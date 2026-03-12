from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

import flet as ft

from hatstand.application.playground_service import PlaygroundService
from hatstand.backends.transformers_backend import TransformersBackend
from hatstand.core.chat_features import available_output_modes
from hatstand.core.model_registry import (
    DEFAULT_MODEL_IDS,
    MODEL_PRESETS,
    find_preset_by_options,
)
from hatstand.core.paths import build_app_paths, ensure_app_paths
from hatstand.core.presets import PROMPT_PRESETS, SYSTEM_PROMPT_PRESETS
from hatstand.domain.dtos import BenchmarkRow, BenchmarkSpec, LoadOptions
from hatstand.domain.entities import (
    AppSettings,
    ChatMessage,
    GenerationParameters,
    ModelStatus,
)
from hatstand.domain.enums import (
    DeviceMap,
    DType,
    MessageRole,
    OutputMode,
    Quantization,
)
from hatstand.infra.logging_setup import configure_logging


CUSTOM_PROMPT_KEY = "__custom__"


class FletPlaygroundApp:
    def __init__(self, page: ft.Page, service: PlaygroundService | None = None) -> None:
        self.page = page
        self.service = service or PlaygroundService()
        configure_logging(self.service.paths.log_path, self.service.settings.log_level)
        self.current_view_index = 0
        self.benchmark_rows: list[BenchmarkRow] = []
        self.selected_load_options = self._load_options_from_session_model()
        self._syncing = False
        self._syncing_prompt = False
        self._chat_input_has_focus = False
        self._chat_shortcut_pending = False
        self._chat_generation_pending = False
        self._chat_scroll_has_overflow = False
        self._chat_surface_hovered = False
        self._chat_input_draft = ""
        self._pending_user_message = ""
        self._preset_name_draft = ""
        self._session_search_query = ""
        self._session_filter_mode = "all"
        self._session_sort_mode = "updated_desc"
        self._active_chat_tool_panel = ""
        self._editing_message_id = ""
        self._editing_message_role = ""
        self._nav_items: list[tuple[ft.Container, ft.Icon, ft.Text]] = []

        self._configure_page()
        self._build_controls()
        self._build_layout()
        self._refresh_all()
        self.page.add(self.root)

    def _configure_page(self) -> None:
        self.page.title = "Hatstand"
        self.page.padding = 0
        self.page.spacing = 0
        self.page.theme_mode = (
            ft.ThemeMode.DARK
            if self.service.settings.theme == "dark"
            else ft.ThemeMode.LIGHT
        )
        self.page.bgcolor = (
            "#0F172A" if self.service.settings.theme == "dark" else "#F8FAFC"
        )
        if self.page.window is not None:
            _icon_path = (
                Path(__file__).resolve().parent.parent.parent / "assets" / "favicon.ico"
            )
            self.page.window.icon = str(_icon_path)
            self.page.window.width = 1540
            self.page.window.height = 980
            self.page.window.min_width = 960
            self.page.window.min_height = 720
        self.page.on_keyboard_event = self._on_page_keyboard_event

    def _build_controls(self) -> None:
        self.clipboard_service = ft.Clipboard()
        if hasattr(self.page, "services"):
            self.page.services.append(self.clipboard_service)
        self.status_text = ft.Text(size=13)

        self.session_dropdown = ft.Dropdown(
            label="Session",
            expand=True,
            options=[],
            on_select=self._on_session_selected,
        )
        self.session_title_field = ft.TextField(
            label="Title",
            expand=True,
            on_change=self._on_session_title_changed,
            on_blur=self._on_session_title_blur,
        )
        self.session_panel_title_text = ft.Text(
            size=16,
            weight=ft.FontWeight.W_600,
            color="#E2E8F0",
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self.session_search_field = ft.TextField(
            label="Find",
            expand=True,
            hint_text="Search title or message",
            on_change=self._on_session_search_changed,
        )
        self.session_filter_dropdown = ft.Dropdown(
            label="Filter",
            expand=True,
            value="all",
            options=[
                ft.dropdown.Option("all", "All"),
                ft.dropdown.Option("pinned", "Pinned"),
                ft.dropdown.Option("empty", "Empty"),
            ],
            on_select=self._on_session_filter_changed,
        )
        self.session_sort_dropdown = ft.Dropdown(
            label="Sort",
            expand=True,
            value="updated_desc",
            options=[
                ft.dropdown.Option("updated_desc", "Recent"),
                ft.dropdown.Option("updated_asc", "Oldest"),
                ft.dropdown.Option("title_asc", "Title"),
            ],
            on_select=self._on_session_sort_changed,
        )
        self.session_new_button = ft.IconButton(
            icon=ft.Icons.ADD_COMMENT,
            tooltip="New Session",
            on_click=self._on_new_chat_clicked,
        )
        self.pin_session_checkbox = ft.Checkbox(
            label="Pinned", on_change=self._on_pin_session_changed
        )
        self.duplicate_session_button = ft.TextButton(
            "Duplicate", on_click=self._on_duplicate_session_clicked
        )
        self.session_list_summary_text = ft.Text(size=12, color="#94A3B8")
        self.session_list_view = ft.ListView(expand=True, spacing=4, auto_scroll=False)
        self.chat_stats_text = ft.Text("Ready.", size=12, color="#94A3B8")
        self.chat_meta_primary_text = ft.Text("", size=11, color="#94A3B8")
        self.chat_meta_secondary_text = ft.Text("", size=11, color="#94A3B8")
        self.chat_messages = ft.ListView(
            expand=True,
            spacing=12,
            auto_scroll=False,
            scroll=ft.ScrollMode.AUTO,
        )
        self.chat_input = ft.TextField(
            label="Message",
            multiline=True,
            min_lines=4,
            max_lines=8,
            expand=True,
            hint_text="Message Hatstand",
            border_width=0,
            filled=False,
            content_padding=ft.Padding.only(left=14, top=14, right=14, bottom=68),
            on_change=self._on_chat_input_changed,
            on_focus=self._on_chat_input_focus,
            on_blur=self._on_chat_input_blur,
        )
        self.output_mode_dropdown = ft.Dropdown(
            label="Output",
            width=170,
            options=[
                ft.dropdown.Option(mode, mode.replace("_", " ").title())
                for mode in available_output_modes()
            ],
            on_select=self._on_output_mode_changed,
        )
        self.thinking_checkbox = ft.Checkbox(
            label="Thinking", on_change=self._on_thinking_changed
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            tooltip="Send",
            on_click=self._on_generate_clicked,
        )
        self.regenerate_button = ft.IconButton(
            icon=ft.Icons.REPLAY,
            tooltip="Regenerate",
            on_click=self._on_regenerate_clicked,
        )
        self.resend_button = ft.IconButton(
            icon=ft.Icons.ARROW_CIRCLE_DOWN_ROUNDED,
            icon_color=ft.Colors.with_opacity(0.72, "#CBD5E1"),
            icon_size=18,
            width=32,
            height=32,
            tooltip="Resend last user message",
            on_click=self._on_resend_clicked,
        )
        self.export_markdown_button = ft.TextButton(
            "Export MD", on_click=self._on_export_session_markdown_clicked
        )
        self.export_json_button = ft.TextButton(
            "Export JSON", on_click=self._on_export_session_json_clicked
        )
        self.message_edit_title = ft.Text(
            "Message Edit", size=14, weight=ft.FontWeight.W_600
        )
        self.message_edit_hint = ft.Text("", size=11, color="#94A3B8")
        self.message_edit_field = ft.TextField(
            label="Selected Message",
            multiline=True,
            min_lines=3,
            max_lines=6,
            expand=True,
        )
        self.apply_edit_button = ft.Button(
            "Apply Edit", on_click=self._on_apply_message_edit_clicked
        )
        self.cancel_edit_button = ft.TextButton(
            "Cancel", on_click=self._on_cancel_message_edit_clicked
        )

        self.temperature_field = ft.TextField(label="Temperature", value="0.7")
        self.top_p_field = ft.TextField(label="Top P", value="0.95")
        self.max_new_tokens_field = ft.TextField(label="Max New Tokens", value="256")
        self.repetition_penalty_field = ft.TextField(
            label="Repetition Penalty", value="1.05"
        )
        self.seed_field = ft.TextField(label="Seed", value="")
        for control in (
            self.session_dropdown,
            self.session_title_field,
            self.session_search_field,
            self.session_filter_dropdown,
            self.session_sort_dropdown,
            self.output_mode_dropdown,
            self.temperature_field,
            self.top_p_field,
            self.max_new_tokens_field,
            self.repetition_penalty_field,
            self.seed_field,
        ):
            self._style_chat_tool_input(control)
        for control in (
            self.pin_session_checkbox,
            self.thinking_checkbox,
        ):
            self._style_chat_tool_checkbox(control)

        self.prompt_preset_dropdown = ft.Dropdown(
            label="Preset",
            expand=True,
            options=[],
            on_select=self._on_prompt_preset_changed,
        )
        self.system_prompt_field = ft.TextField(
            label="Prompt",
            multiline=True,
            min_lines=12,
            max_lines=20,
            expand=True,
            on_change=self._on_system_prompt_changed,
            on_blur=self._on_system_prompt_blur,
        )
        self.preset_name_field = ft.TextField(
            label="New Preset Name",
            expand=True,
            on_change=self._on_preset_name_changed,
            on_blur=self._on_preset_name_blur,
        )
        self.create_preset_button = ft.Button(
            "SAVE", on_click=self._on_create_prompt_preset
        )
        self.delete_preset_button = ft.TextButton(
            "Delete Selected", on_click=self._on_delete_prompt_preset
        )

        self.model_preset_dropdown = ft.Dropdown(
            label="Preset",
            options=[ft.dropdown.Option(preset.name) for preset in MODEL_PRESETS],
            on_select=self._on_model_preset_changed,
        )
        self.model_id_dropdown = ft.TextField(
            label="Model ID",
            hint_text="e.g. Qwen/Qwen3.5-2B or HF URL",
            on_change=self._on_model_options_changed,
        )
        self.quant_dropdown = ft.Dropdown(
            label="Quantization",
            options=[ft.dropdown.Option(item.value) for item in Quantization],
            on_select=self._on_model_options_changed,
        )
        self.dtype_dropdown = ft.Dropdown(
            label="DType",
            options=[ft.dropdown.Option(item.value) for item in DType],
            on_select=self._on_model_options_changed,
        )
        self.device_map_dropdown = ft.Dropdown(
            label="Device Map",
            options=[ft.dropdown.Option(item.value) for item in DeviceMap],
            on_select=self._on_model_options_changed,
        )
        self.revision_field = ft.TextField(
            label="Revision",
            hint_text="main (default)",
        )
        self.model_info_text = ft.Text(size=12, color="#94A3B8")
        self.runtime_info_text = ft.Text(size=12, color="#FBBF24")
        self.quantization_hint_text = ft.Text(size=12, color="#94A3B8")
        self.path_info_text = ft.Text(size=12, color="#94A3B8")
        self.load_button = ft.Button(
            "Load", icon=ft.Icons.DOWNLOAD, on_click=self._on_load_model_clicked
        )
        self.unload_button = ft.OutlinedButton(
            "Unload", icon=ft.Icons.CLOSE, on_click=self._on_unload_model_clicked
        )
        self.warmup_button = ft.TextButton("Warmup", on_click=self._on_warmup_clicked)

        self.benchmark_prompt_dropdown = ft.Dropdown(
            label="Prompt",
            options=[ft.dropdown.Option(name) for name in PROMPT_PRESETS],
            value=next(iter(PROMPT_PRESETS.keys()), None),
        )
        self.benchmark_run_button = ft.Button(
            "Run Benchmark",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run_benchmark_clicked,
        )
        self.benchmark_export_button = ft.OutlinedButton(
            "Export CSV",
            icon=ft.Icons.SAVE_ALT,
            on_click=self._on_export_benchmark_clicked,
        )
        self.benchmark_progress_text = ft.Text("Idle.", size=12, color="#94A3B8")
        self.benchmark_model_checks = {
            preset.name: ft.Checkbox(label=preset.name) for preset in MODEL_PRESETS
        }
        self.benchmark_results_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Model")),
                ft.DataColumn(ft.Text("Load")),
                ft.DataColumn(ft.Text("TTFT")),
                ft.DataColumn(ft.Text("Total")),
                ft.DataColumn(ft.Text("Tokens")),
                ft.DataColumn(ft.Text("Tok/s")),
                ft.DataColumn(ft.Text("Success")),
            ],
            rows=[],
        )

        self.hf_cache_field = ft.TextField(label="HF Cache Dir")
        self.app_data_field = ft.TextField(label="App Data Path")
        self.log_level_dropdown = ft.Dropdown(
            label="Log Level",
            options=[
                ft.dropdown.Option(item)
                for item in ["DEBUG", "INFO", "WARNING", "ERROR"]
            ],
        )
        self.default_model_dropdown = ft.Dropdown(
            label="Default Model",
            options=[ft.dropdown.Option(preset.name) for preset in MODEL_PRESETS],
        )
        self.theme_dropdown = ft.Dropdown(
            label="Theme",
            options=[ft.dropdown.Option(item) for item in ["light", "dark"]],
        )
        self.save_settings_button = ft.Button(
            "Save Settings",
            icon=ft.Icons.SAVE,
            on_click=self._on_save_settings_clicked,
        )

        self.chat_page = self._build_chat_page()
        self.prompt_page = self._build_prompt_page()
        self.model_page = self._build_model_page()
        self.benchmark_page = self._build_benchmark_page()
        self.settings_page = self._build_settings_page()
        self.view_pages = [
            self.chat_page,
            self.prompt_page,
            self.model_page,
            self.benchmark_page,
            self.settings_page,
        ]

    def _style_chat_tool_input(self, control: ft.Control) -> None:
        control.bgcolor = ft.Colors.TRANSPARENT
        control.color = "#E2E8F0"
        control.border_color = ft.Colors.with_opacity(0.34, "#CBD5E1")
        control.focused_border_color = "#F59E0B"
        control.label_style = ft.TextStyle(color="#94A3B8")
        if hasattr(control, "hint_style"):
            control.hint_style = ft.TextStyle(color="#64748B")
        if hasattr(control, "text_style"):
            control.text_style = ft.TextStyle(color="#E2E8F0")
        if hasattr(control, "border_radius"):
            control.border_radius = 12

    def _style_chat_tool_checkbox(self, control: ft.Checkbox) -> None:
        control.label_style = ft.TextStyle(color="#E2E8F0")
        control.fill_color = "#F59E0B"
        control.check_color = "#0F172A"
        control.active_color = "#F59E0B"

    def _build_chat_glass_shell(
        self,
        content: ft.Control,
        *,
        padding: int | ft.Padding = 8,
        border_radius: int = 16,
        background_opacity: float = 0.20,
    ) -> ft.Container:
        return ft.Container(
            border_radius=border_radius,
            bgcolor=ft.Colors.with_opacity(background_opacity, "#0F172A"),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.26, "#E2E8F0")),
            blur=ft.Blur(16, 16),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            padding=padding,
            content=content,
        )

    def _build_layout(self) -> None:
        self.content_host = ft.Container(
            content=self.view_pages[self.current_view_index], expand=True
        )
        sidebar = self._build_sidebar()
        status_bar = ft.Container(
            padding=ft.Padding.symmetric(horizontal=20, vertical=12),
            bgcolor="#111827",
            content=self.status_text,
        )
        self.root = ft.Container(
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=0,
                controls=[
                    ft.Row(
                        expand=True,
                        spacing=0,
                        controls=[
                            sidebar,
                            ft.Container(
                                expand=True,
                                padding=20,
                                content=self.content_host,
                            ),
                        ],
                    ),
                    status_bar,
                ],
            ),
        )

    def _build_sidebar(self) -> ft.Control:
        nav_specs = [
            ("Chat", ft.Icons.CHAT_BUBBLE_OUTLINE),
            ("System Prompt", ft.Icons.AUTO_AWESOME_OUTLINED),
            ("Models", ft.Icons.DATASET_OUTLINED),
            ("Benchmark", ft.Icons.INSIGHTS_OUTLINED),
            ("Settings", ft.Icons.SETTINGS_OUTLINED),
        ]
        nav_controls: list[ft.Control] = []
        for index, (label, icon_name) in enumerate(nav_specs):
            icon = ft.Icon(icon_name, color="#94A3B8", size=18)
            text = ft.Text(label, color="#E2E8F0", size=14, weight=ft.FontWeight.W_500)
            card = ft.Container(
                border_radius=14,
                padding=ft.Padding.symmetric(horizontal=14, vertical=12),
                bgcolor="transparent",
                on_click=lambda e, idx=index: self._set_view(idx),
                content=ft.Row(
                    spacing=12,
                    controls=[icon, text],
                ),
            )
            self._nav_items.append((card, icon, text))
            nav_controls.append(card)
        return ft.Container(
            width=250,
            bgcolor="#020617",
            padding=ft.Padding.only(left=18, top=22, right=18, bottom=18),
            content=ft.Column(
                expand=True,
                spacing=18,
                controls=[
                    ft.Row(
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Image(
                                src=str(
                                    Path(__file__).resolve().parent.parent.parent
                                    / "assets"
                                    / "favicon.ico"
                                ),
                                width=40,
                                height=40,
                            ),
                            ft.Column(
                                spacing=2,
                                controls=[
                                    ft.Text(
                                        "Hatstand",
                                        size=28,
                                        weight=ft.FontWeight.W_700,
                                        color="#F8FAFC",
                                    ),
                                    ft.Text(
                                        "Local playground on Flet",
                                        size=13,
                                        color="#94A3B8",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    ft.Button(
                        "New Chat",
                        icon=ft.Icons.ADD_COMMENT,
                        on_click=self._on_new_chat_clicked,
                    ),
                    ft.Column(spacing=8, controls=nav_controls, expand=True),
                    ft.Container(
                        alignment=ft.Alignment(0, 0),
                        on_click=self._on_sponsor_clicked,
                        content=ft.Image(
                            src=str(
                                Path(__file__).resolve().parent.parent.parent
                                / "assets"
                                / "sponsor_logo.png"
                            ),
                            width=150,
                        ),
                    ),
                ],
            ),
        )

    async def _on_sponsor_clicked(self, _: ft.ControlEvent) -> None:
        await self.page.launch_url(
            "https://www.patreon.com/cw/NakamuraShippo/membership"
        )

    def _build_card(self, title: str, subtitle: str, content: ft.Control) -> ft.Control:
        return ft.Container(
            expand=True,
            border_radius=22,
            border=ft.Border.all(1, "#334155"),
            bgcolor="#111827" if self.service.settings.theme == "dark" else "#FFFFFF",
            padding=22,
            content=ft.Column(
                expand=True,
                spacing=18,
                controls=[
                    ft.Column(
                        spacing=4,
                        controls=[
                            ft.Text(title, size=26, weight=ft.FontWeight.W_700),
                            ft.Text(subtitle, size=13, color="#94A3B8"),
                        ],
                    ),
                    content,
                ],
            ),
        )

    def _build_chat_page(self) -> ft.Control:
        self.chat_tool_panel_title = ft.Text(size=18, weight=ft.FontWeight.W_600)
        self.chat_tool_panel_subtitle = ft.Text(size=12, color="#94A3B8")
        self.chat_tool_session_content = ft.Column(
            spacing=12,
            visible=False,
            controls=[
                ft.Row(
                    wrap=False,
                    spacing=10,
                    controls=[self.session_filter_dropdown, self.session_sort_dropdown],
                ),
                self.session_list_summary_text,
                self._build_chat_glass_shell(
                    ft.Container(
                        height=580,
                        padding=2,
                        content=self.session_list_view,
                    ),
                    padding=10,
                    border_radius=14,
                    background_opacity=0.18,
                ),
            ],
        )
        self.chat_tool_output_content = ft.Column(
            spacing=12,
            visible=False,
            controls=[
                self.output_mode_dropdown,
                self.thinking_checkbox,
                ft.Text(
                    "Choose the output shape here. Copy and export actions stay near the composer.",
                    size=12,
                    color="#94A3B8",
                ),
            ],
        )
        self.chat_tool_controls_content = ft.Column(
            spacing=12,
            visible=False,
            controls=[
                self._build_chat_glass_shell(self.temperature_field),
                self._build_chat_glass_shell(self.top_p_field),
                self._build_chat_glass_shell(self.max_new_tokens_field),
                self._build_chat_glass_shell(self.repetition_penalty_field),
                self._build_chat_glass_shell(self.seed_field),
                ft.Text(
                    "Leave seed empty for random generation.", size=12, color="#94A3B8"
                ),
            ],
        )
        self.chat_tool_panel_body = ft.Column(
            spacing=12,
            controls=[
                self.chat_tool_session_content,
                self.chat_tool_output_content,
                self.chat_tool_controls_content,
            ],
        )
        self.chat_tool_panel = ft.Container(
            width=340,
            visible=False,
            left=0,
            top=56,
            padding=16,
            border_radius=18,
            bgcolor=ft.Colors.with_opacity(0.34, "#0F172A"),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.30, "#E2E8F0")),
            blur=ft.Blur(18, 18),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Column(
                spacing=12,
                controls=[
                    self.chat_tool_panel_title,
                    self.chat_tool_panel_subtitle,
                    self.chat_tool_panel_body,
                ],
            ),
        )
        self.chat_tool_tabs: dict[str, tuple[ft.Container, ft.Text]] = {}
        tab_rail = ft.Container(
            width=28,
            margin=ft.Margin.only(top=72),
            content=ft.Column(
                spacing=4,
                controls=[
                    self._build_chat_tool_tab("session", "Session"),
                    self._build_chat_tool_tab("controls", "Controls"),
                    self._build_chat_tool_tab("output", "Output"),
                ],
            ),
        )
        status_card = ft.Container(
            padding=ft.Padding.symmetric(horizontal=14, vertical=12),
            border_radius=18,
            bgcolor=ft.Colors.with_opacity(0.34, "#0F172A"),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.30, "#E2E8F0")),
            blur=ft.Blur(18, 18),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Column(
                spacing=4,
                controls=[
                    self.chat_stats_text,
                    self.chat_meta_primary_text,
                    self.chat_meta_secondary_text,
                ],
            ),
        )
        edit_container = ft.Container(
            visible=False,
            padding=14,
            border_radius=14,
            bgcolor="#172554" if self.service.settings.theme == "dark" else "#DBEAFE",
            content=ft.Column(
                spacing=10,
                controls=[
                    self.message_edit_title,
                    self.message_edit_hint,
                    self.message_edit_field,
                    ft.Row(
                        alignment=ft.MainAxisAlignment.END,
                        controls=[self.cancel_edit_button, self.apply_edit_button],
                    ),
                ],
            ),
        )
        self.message_edit_container = edit_container
        self.chat_resend_overlay = ft.Container(
            visible=False,
            right=12,
            bottom=18,
            bgcolor=ft.Colors.with_opacity(0.52, "#334155"),
            border_radius=999,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.38, "#94A3B8")),
            content=self.resend_button,
        )
        self.chat_tool_dismiss_overlay = ft.Container(
            visible=False,
            left=0,
            right=0,
            top=0,
            bottom=0,
            bgcolor=ft.Colors.with_opacity(0.001, "#020617"),
            on_click=self._on_chat_tool_dismiss_clicked,
        )
        chat_header_overlay = ft.Container(
            left=0,
            right=0,
            top=0,
            padding=ft.Padding.only(top=2),
            ignore_interactions=True,
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.START,
                controls=[
                    ft.Text("Chat", size=26, weight=ft.FontWeight.W_700),
                    status_card,
                ],
            ),
        )
        composer = ft.Column(
            spacing=12,
            controls=[
                ft.Container(
                    border_radius=18,
                    bgcolor=(
                        "#0F172A"
                        if self.service.settings.theme == "dark"
                        else "#F8FAFC"
                    ),
                    border=ft.Border.all(1, "#1E293B"),
                    padding=0,
                    content=ft.Stack(
                        controls=[
                            self.chat_input,
                            ft.Container(
                                right=14,
                                bottom=14,
                                content=ft.Row(
                                    tight=True,
                                    spacing=8,
                                    controls=[
                                        self.session_new_button,
                                        self.regenerate_button,
                                        self.send_button,
                                    ],
                                ),
                            ),
                        ],
                    ),
                ),
                self.message_edit_container,
            ],
        )
        main_chat = ft.Column(
            expand=True,
            spacing=14,
            controls=[
                ft.Container(
                    expand=True,
                    border_radius=18,
                    bgcolor=(
                        "rgba(15,23,42,0.55)"
                        if self.service.settings.theme == "dark"
                        else "rgba(248,250,252,0.55)"
                    ),
                    blur=ft.Blur(12, 12, ft.BlurTileMode.MIRROR),
                    padding=18,
                    on_hover=self._on_chat_messages_hover,
                    content=ft.Stack(
                        expand=True,
                        controls=[
                            self.chat_messages,
                            self.chat_resend_overlay,
                        ],
                    ),
                ),
                composer,
            ],
        )
        self.chat_page_stack = ft.Stack(
            expand=True,
            controls=[
                main_chat,
                self.chat_tool_dismiss_overlay,
                self.chat_tool_panel,
                chat_header_overlay,
            ],
        )
        chat_box = ft.Container(
            expand=True,
            border_radius=22,
            border=ft.Border.all(1, "#334155"),
            bgcolor=(
                "rgba(17,24,39,0.50)"
                if self.service.settings.theme == "dark"
                else "rgba(255,255,255,0.50)"
            ),
            blur=ft.Blur(14, 14, ft.BlurTileMode.MIRROR),
            padding=22,
            content=self.chat_page_stack,
        )
        return ft.Row(
            expand=True,
            spacing=0,
            controls=[tab_rail, chat_box],
        )

    def _build_chat_tool_tab(self, key: str, label: str) -> ft.Control:
        text = ft.Text(
            label,
            size=12,
            weight=ft.FontWeight.W_600,
            color="#FEF3C7",
            text_align=ft.TextAlign.CENTER,
            no_wrap=True,
        )
        # Use Stack so the rotated inner escapes the 28px width constraint.
        inner = ft.Container(
            width=100,
            height=28,
            alignment=ft.Alignment(0, 0),
            rotate=ft.Rotate(angle=-1.5708),
            left=-36,
            top=36,
            content=text,
        )
        card = ft.Container(
            width=28,
            height=100,
            border_radius=ft.BorderRadius.only(top_left=8, bottom_left=8),
            bgcolor="#B45309",
            clip_behavior=ft.ClipBehavior.NONE,
            on_click=lambda e, panel_key=key: self._on_chat_tool_tab_clicked(panel_key),
            content=ft.Stack(
                controls=[inner],
            ),
        )
        self.chat_tool_tabs[key] = (card, text)
        return card

    def _refresh_chat_tool_panel(self) -> None:
        self.chat_tool_panel.visible = bool(self._active_chat_tool_panel)
        self.chat_tool_dismiss_overlay.visible = bool(self._active_chat_tool_panel)
        panel_copy = {
            "session": (
                "Session Stack",
                "Switch, search, and organize sessions from a dedicated side drawer.",
            ),
            "output": (
                "Output Options",
                "Control formatting and reasoning display for the next reply.",
            ),
            "controls": (
                "Generation Controls",
                "Tune sampling and token limits without leaving the chat.",
            ),
        }
        title, subtitle = panel_copy.get(self._active_chat_tool_panel, ("", ""))
        self.chat_tool_panel_title.value = title
        self.chat_tool_panel_subtitle.value = subtitle
        self.chat_tool_session_content.visible = (
            self._active_chat_tool_panel == "session"
        )
        self.chat_tool_output_content.visible = self._active_chat_tool_panel == "output"
        self.chat_tool_controls_content.visible = (
            self._active_chat_tool_panel == "controls"
        )
        for key, (card, text) in self.chat_tool_tabs.items():
            is_active = key == self._active_chat_tool_panel
            card.bgcolor = "#92400E" if is_active else "#B45309"
            text.color = "#FFFFFF" if is_active else "#FEF3C7"

    def _build_session_list_item(
        self, session: object, is_current: bool, busy: bool
    ) -> ft.Control:
        session_id = str(getattr(session, "session_id", ""))
        session_title = str(getattr(session, "title", "Untitled Session"))
        is_pinned = bool(getattr(session, "pinned", False))
        meta_parts = [
            self._format_message_timestamp(str(getattr(session, "updated_at", "")))
        ]
        message_count = len(getattr(session, "messages", []))
        meta_parts.append(f"{message_count} msg" if message_count != 1 else "1 msg")
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.28 if is_current else 0.18, "#0F172A"),
            border=ft.Border.all(
                1,
                (
                    ft.Colors.with_opacity(0.90, "#F59E0B")
                    if is_current
                    else ft.Colors.with_opacity(0.20, "#E2E8F0")
                ),
            ),
            blur=ft.Blur(16, 16),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            ink=not busy,
            on_click=(
                None
                if busy or is_current
                else lambda e, sid=str(
                    getattr(session, "session_id", "")
                ): self._activate_session(sid)
            ),
            content=ft.Column(
                spacing=2,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Row(
                                expand=True,
                                spacing=8,
                                controls=[
                                    ft.Text(
                                        session_title,
                                        weight=ft.FontWeight.W_600,
                                        color="#F8FAFC" if is_current else "#E2E8F0",
                                        max_lines=1,
                                        overflow=ft.TextOverflow.ELLIPSIS,
                                        expand=True,
                                    ),
                                ],
                            ),
                            self._build_session_actions_menu(
                                session_id, session_title, is_pinned, busy
                            ),
                        ],
                    ),
                    ft.Text(
                        " | ".join(part for part in meta_parts if part),
                        size=10,
                        color="#64748B",
                    ),
                ],
            ),
        )

    def _build_session_actions_menu(
        self, session_id: str, session_title: str, is_pinned: bool, busy: bool
    ) -> ft.Control:
        return ft.PopupMenuButton(
            icon=ft.Icons.MENU,
            icon_color="#94A3B8",
            tooltip="Session actions",
            disabled=busy,
            items=[
                ft.PopupMenuItem(
                    "Pinned",
                    icon=ft.Icons.PUSH_PIN_OUTLINED,
                    checked=is_pinned,
                    on_click=lambda e, sid=session_id, pinned=is_pinned: self._on_session_pin_menu_clicked(
                        sid, pinned
                    ),
                ),
                ft.PopupMenuItem(),
                ft.PopupMenuItem(
                    "Duplicate",
                    icon=ft.Icons.CONTENT_COPY,
                    on_click=lambda e, sid=session_id: self._on_session_duplicate_menu_clicked(
                        sid
                    ),
                ),
                ft.PopupMenuItem(
                    "Delete",
                    icon=ft.Icons.DELETE_OUTLINE,
                    on_click=lambda e, sid=session_id: self._on_session_delete_menu_clicked(
                        sid
                    ),
                ),
                ft.PopupMenuItem(),
                ft.PopupMenuItem(
                    "Export MD",
                    icon=ft.Icons.DESCRIPTION_OUTLINED,
                    on_click=lambda e, sid=session_id, title=session_title: self._on_session_export_markdown_menu_clicked(
                        sid, title
                    ),
                ),
                ft.PopupMenuItem(
                    "Export JSON",
                    icon=ft.Icons.DATA_OBJECT,
                    on_click=lambda e, sid=session_id, title=session_title: self._on_session_export_json_menu_clicked(
                        sid, title
                    ),
                ),
            ],
        )

    def _build_prompt_page(self) -> ft.Control:
        return self._build_card(
            "System Prompt",
            "Choose or edit the instruction that guides the assistant before every reply.",
            ft.Column(
                expand=True,
                spacing=14,
                controls=[
                    ft.Row(controls=[self.prompt_preset_dropdown]),
                    ft.Row(controls=[self.system_prompt_field]),
                    ft.Row(controls=[self.preset_name_field]),
                    ft.Row(
                        spacing=10,
                        alignment=ft.MainAxisAlignment.END,
                        controls=[self.delete_preset_button, self.create_preset_button],
                    ),
                    ft.Text(
                        "The active prompt is stored on the current chat session and becomes the default for new chats.",
                        size=12,
                        color="#94A3B8",
                    ),
                ],
            ),
        )

    def _build_model_page(self) -> ft.Control:
        return self._build_card(
            "Model Workspace",
            "Switch local models, quantization, and runtime behavior without leaving the app.",
            ft.Column(
                spacing=14,
                controls=[
                    self.model_preset_dropdown,
                    self.model_id_dropdown,
                    self.revision_field,
                    self.quant_dropdown,
                    self.dtype_dropdown,
                    self.device_map_dropdown,
                    ft.Row(
                        wrap=True,
                        spacing=10,
                        controls=[
                            self.load_button,
                            self.unload_button,
                            self.warmup_button,
                        ],
                    ),
                    ft.Text(
                        "Warmup runs a short readiness prompt and primes the model cache.",
                        size=12,
                        color="#94A3B8",
                    ),
                    self.model_info_text,
                    self.runtime_info_text,
                    self.quantization_hint_text,
                    self.path_info_text,
                ],
            ),
        )

    def _build_benchmark_page(self) -> ft.Control:
        return self._build_card(
            "Benchmark Lab",
            "Compare presets, run repeatable prompts, and export the results.",
            ft.Column(
                spacing=16,
                controls=[
                    ft.Row(
                        wrap=True,
                        spacing=12,
                        controls=[
                            self.benchmark_prompt_dropdown,
                            self.benchmark_run_button,
                            self.benchmark_export_button,
                        ],
                    ),
                    ft.Text("Models", size=18, weight=ft.FontWeight.W_600),
                    ft.Column(
                        spacing=6,
                        controls=list(self.benchmark_model_checks.values()),
                    ),
                    self.benchmark_progress_text,
                    ft.Container(
                        border_radius=18,
                        bgcolor=(
                            "#0F172A"
                            if self.service.settings.theme == "dark"
                            else "#F8FAFC"
                        ),
                        padding=12,
                        content=ft.Column(
                            spacing=8,
                            controls=[
                                ft.Text("Results", size=18, weight=ft.FontWeight.W_600),
                                ft.Row(
                                    scroll=ft.ScrollMode.AUTO,
                                    controls=[self.benchmark_results_table],
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        )

    def _build_settings_page(self) -> ft.Control:
        return self._build_card(
            "Workspace Settings",
            "Tune cache paths, defaults, and how the local playground feels.",
            ft.Column(
                spacing=14,
                controls=[
                    self.hf_cache_field,
                    self.app_data_field,
                    self.log_level_dropdown,
                    self.default_model_dropdown,
                    self.theme_dropdown,
                    self.save_settings_button,
                ],
            ),
        )

    def _set_view(self, index: int) -> None:
        self.current_view_index = index
        self.content_host.content = self.view_pages[index]
        self._refresh_nav_state()
        self.page.update()

    def _refresh_nav_state(self) -> None:
        for index, (card, icon, text) in enumerate(self._nav_items):
            selected = index == self.current_view_index
            card.bgcolor = "#B45309" if selected else "transparent"
            icon.color = "#FEF3C7" if selected else "#94A3B8"
            text.color = "#FEF3C7" if selected else "#E2E8F0"

    def _refresh_all(self) -> None:
        self._syncing = True
        self._refresh_nav_state()
        self._refresh_chat_controls()
        self._refresh_prompt_controls()
        self._refresh_model_controls()
        self._refresh_benchmark_controls()
        self._refresh_settings_controls()
        self._refresh_status_bar()
        self._syncing = False

    def _refresh_chat_controls(self) -> None:
        self._session_search_query = ""
        sessions = self.service.list_sessions(query="", sort_by=self._session_sort_mode)
        if self._session_filter_mode == "pinned":
            sessions = [session for session in sessions if session.pinned]
        elif self._session_filter_mode == "empty":
            sessions = [session for session in sessions if not session.messages]
        current_id = self.service.current_session.session_id
        if not any(session.session_id == current_id for session in sessions):
            should_show_current = not self._session_search_query
            if self._session_filter_mode == "pinned":
                should_show_current = self.service.current_session.pinned
            elif self._session_filter_mode == "empty":
                should_show_current = not self.service.current_session.messages
            if should_show_current:
                sessions = [self.service.current_session, *sessions]
        self.session_dropdown.options = [ft.dropdown.Option("", "(New Session)")]
        self.session_dropdown.options.extend(
            ft.dropdown.Option(
                session.session_id, self._build_session_option_label(session)
            )
            for session in sessions
        )
        if any(session.session_id == current_id for session in sessions):
            self.session_dropdown.value = current_id
        else:
            self.session_dropdown.value = ""
        self.session_title_field.value = self.service.current_session.title
        self.session_panel_title_text.value = (
            self.service.current_session.title or "Untitled Session"
        )
        self.session_search_field.value = ""
        self.session_filter_dropdown.value = self._session_filter_mode
        self.session_sort_dropdown.value = self._session_sort_mode
        self.pin_session_checkbox.value = self.service.current_session.pinned
        params = self.service.current_session.generation
        self.temperature_field.value = str(params.temperature)
        self.top_p_field.value = str(params.top_p)
        self.max_new_tokens_field.value = str(params.max_new_tokens)
        self.repetition_penalty_field.value = str(params.repetition_penalty)
        self.seed_field.value = "" if params.seed is None else str(params.seed)
        self.output_mode_dropdown.value = (
            self.service.current_session.output_mode or OutputMode.NORMAL.value
        )
        self.thinking_checkbox.value = self.service.current_session.thinking_enabled
        visible_messages = list(self.service.current_session.messages)
        if self._pending_user_message and not self._pending_user_message_is_committed(
            visible_messages
        ):
            visible_messages.append(
                ChatMessage(role="user", content=self._pending_user_message)
            )
        self.chat_messages.controls = [
            self._build_message_bubble(message) for message in visible_messages
        ]
        self._chat_scroll_has_overflow = self._estimate_chat_scroll_overflow(
            visible_messages
        )

        status = self.service.status
        generation_busy = self._chat_generation_pending or status.generation_state in {
            "GENERATING",
            "STOPPING",
        }
        model_busy = status.model_state in {"LOADING", "UNLOADING"}
        busy = generation_busy or model_busy
        self.send_button.icon = ft.Icons.STOP if generation_busy else ft.Icons.SEND
        self.send_button.tooltip = "Stop" if generation_busy else "Send"
        self.send_button.on_click = (
            self._on_stop_clicked if generation_busy else self._on_generate_clicked
        )
        self.send_button.disabled = model_busy or status.generation_state == "STOPPING"
        self.regenerate_button.disabled = busy or self._last_assistant_message() is None
        self.resend_button.disabled = busy or self._last_user_message() is None
        self._update_chat_resend_overlay_visibility()
        self.export_markdown_button.disabled = not bool(
            self.service.current_session.messages
        )
        self.export_json_button.disabled = False
        self.chat_input.disabled = busy and status.generation_state != "STOPPING"
        self.session_dropdown.disabled = busy
        self.session_filter_dropdown.disabled = busy
        self.session_title_field.disabled = busy
        self.session_search_field.disabled = busy
        self.session_sort_dropdown.disabled = busy
        self.session_new_button.disabled = busy
        self.pin_session_checkbox.disabled = busy
        self.duplicate_session_button.disabled = busy
        self.output_mode_dropdown.disabled = busy
        self.thinking_checkbox.disabled = busy
        self.message_edit_field.disabled = busy
        self.apply_edit_button.disabled = busy or not bool(
            (self.message_edit_field.value or "").strip()
        )
        self.cancel_edit_button.disabled = busy
        for field in (
            self.temperature_field,
            self.top_p_field,
            self.max_new_tokens_field,
            self.repetition_penalty_field,
            self.seed_field,
        ):
            field.disabled = busy
        self.chat_stats_text.value = self._build_chat_stats_text(status)
        meta_primary, meta_secondary = self._build_chat_meta_text()
        self.chat_meta_primary_text.value = meta_primary
        self.chat_meta_secondary_text.value = meta_secondary
        self.session_list_summary_text.value = "Sessions"
        if sessions:
            self.session_list_view.controls = [
                self._build_session_list_item(
                    session, session.session_id == current_id, busy
                )
                for session in sessions
            ]
        else:
            self.session_list_view.controls = [
                ft.Container(
                    padding=16,
                    border_radius=14,
                    bgcolor="#0B1220",
                    content=ft.Text(
                        "No sessions match the current search or filter.",
                        size=12,
                        color="#94A3B8",
                    ),
                )
            ]
        self._refresh_chat_tool_panel()
        self.message_edit_container.visible = bool(self._editing_message_id)
        if self._editing_message_id:
            action_label = (
                "Apply + Rerun"
                if self._editing_message_role == MessageRole.USER.value
                else "Apply Edit"
            )
            self.apply_edit_button.content = action_label
            self.message_edit_hint.value = (
                "Editing a user message will rebuild the conversation from that point."
                if self._editing_message_role == MessageRole.USER.value
                else "Editing a non-user message trims later replies and saves the change."
            )
        else:
            self.apply_edit_button.content = "Apply Edit"
            self.message_edit_hint.value = ""

    def _refresh_prompt_controls(
        self,
        sync_prompt_field: bool = True,
        sync_preset_name_field: bool = True,
    ) -> None:
        prompt_options = [ft.dropdown.Option(CUSTOM_PROMPT_KEY, "(Custom)")]
        prompt_options.extend(
            ft.dropdown.Option(name, name) for name in SYSTEM_PROMPT_PRESETS
        )
        prompt_options.extend(
            ft.dropdown.Option(name, name)
            for name in self.service.settings.system_prompt_presets
        )
        current_prompt = self.service.current_session.system_prompt
        self.prompt_preset_dropdown.options = prompt_options
        self.prompt_preset_dropdown.value = self._find_matching_prompt_key(
            current_prompt
        )
        if sync_prompt_field:
            self.system_prompt_field.value = current_prompt
        selected_name = self.prompt_preset_dropdown.value or CUSTOM_PROMPT_KEY
        if sync_preset_name_field:
            if selected_name in self.service.settings.system_prompt_presets:
                self.preset_name_field.value = selected_name
            elif selected_name == CUSTOM_PROMPT_KEY:
                draft_name = (
                    self._preset_name_draft or self.preset_name_field.value or ""
                )
                self.preset_name_field.value = (
                    ""
                    if draft_name not in self.service.settings.system_prompt_presets
                    else draft_name
                )
            else:
                self.preset_name_field.value = ""
            self._preset_name_draft = self.preset_name_field.value or ""

        enabled = not (
            self.service.status.generation_state in {"GENERATING", "STOPPING"}
            or self.service.status.model_state in {"LOADING", "UNLOADING"}
        )
        self.prompt_preset_dropdown.disabled = not enabled
        self.system_prompt_field.disabled = not enabled
        self.preset_name_field.disabled = not enabled
        self.create_preset_button.disabled = not (
            enabled
            and bool(
                (self._preset_name_draft or self.preset_name_field.value or "").strip()
            )
            and bool(self.service.current_session.system_prompt.strip())
        )
        self.delete_preset_button.disabled = not (
            enabled
            and self.prompt_preset_dropdown.value
            in self.service.settings.system_prompt_presets
        )

    def _refresh_model_controls(self) -> None:
        options = self.selected_load_options
        self.model_id_dropdown.value = options.model_id or DEFAULT_MODEL_IDS[0]
        self.quant_dropdown.value = options.quant or Quantization.NONE.value
        self.dtype_dropdown.value = options.dtype or DType.BFLOAT16.value
        self.device_map_dropdown.value = options.device_map or DeviceMap.AUTO.value
        preset = find_preset_by_options(
            self.model_id_dropdown.value or "",
            self.quant_dropdown.value or "",
            self.dtype_dropdown.value or "",
            self.device_map_dropdown.value or "",
        )
        self.model_preset_dropdown.value = preset.name if preset is not None else None
        status = self.service.status
        operation_busy = status.model_state in {
            "LOADING",
            "UNLOADING",
        } or status.generation_state in {
            "GENERATING",
            "STOPPING",
        }
        self.load_button.disabled = operation_busy
        self.unload_button.disabled = status.model_state != "LOADED" or operation_busy
        self.warmup_button.disabled = status.model_state != "LOADED" or operation_busy
        self.model_preset_dropdown.disabled = operation_busy
        self.model_id_dropdown.disabled = operation_busy
        self.revision_field.disabled = operation_busy
        self.quant_dropdown.disabled = operation_busy
        self.dtype_dropdown.disabled = operation_busy
        self.device_map_dropdown.disabled = operation_busy
        self.model_info_text.value = f"Current model: {status.current_model_id or '(unloaded)'} | state={status.model_state}"
        self.runtime_info_text.value = self._build_runtime_info_text()
        quant_hint, quant_hint_color = self._build_quantization_hint_text()
        self.quantization_hint_text.value = quant_hint
        self.quantization_hint_text.color = quant_hint_color
        cache_value = (
            self.service.settings.hf_cache_dir or "(default Hugging Face cache)"
        )
        self.path_info_text.value = (
            f"Cache: {cache_value} | Sessions: {self.service.paths.sessions_dir}"
        )

    def _refresh_benchmark_controls(self) -> None:
        running = (
            self.service.status.model_state == "LOADING"
            and self.service.status.generation_state == "IDLE"
        )
        self.benchmark_prompt_dropdown.disabled = running
        self.benchmark_run_button.disabled = running
        self.benchmark_export_button.disabled = running or not self.benchmark_rows
        for checkbox in self.benchmark_model_checks.values():
            checkbox.disabled = running
        self.benchmark_results_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(row.model_name)),
                    ft.DataCell(ft.Text(f"{row.load_seconds:.2f}")),
                    ft.DataCell(ft.Text(f"{row.first_token_seconds:.2f}")),
                    ft.DataCell(ft.Text(f"{row.total_seconds:.2f}")),
                    ft.DataCell(ft.Text(str(row.output_tokens))),
                    ft.DataCell(ft.Text(f"{row.tokens_per_second:.2f}")),
                    ft.DataCell(ft.Text("yes" if row.success else "no")),
                ]
            )
            for row in self.benchmark_rows
        ]

    def _refresh_settings_controls(self) -> None:
        settings = self.service.settings
        self.hf_cache_field.value = settings.hf_cache_dir
        self.app_data_field.value = settings.app_data_path
        self.log_level_dropdown.value = settings.log_level
        self.default_model_dropdown.value = settings.default_model_preset
        self.theme_dropdown.value = settings.theme

    def _refresh_status_bar(self) -> None:
        status = self.service.status
        summary = f"{status.model_state} / {status.generation_state}"
        if status.current_model_id:
            summary = f"{summary} | {status.current_model_id}"
        if status.error_message:
            summary = f"{summary} | {status.error_message}"
        self.status_text.value = summary
        self.status_text.color = "#F8FAFC"

    def _build_message_bubble(self, message: ChatMessage) -> ft.Control:
        is_user = message.role == MessageRole.USER.value
        label = "You" if is_user else "Assistant"
        background = "#B45309" if is_user else "rgba(30,41,59,0.55)"
        content_control = self._build_message_content_control(message)
        action_controls = [
            ft.TextButton(
                "Edit",
                on_click=lambda e, mid=message.message_id: self._on_edit_message_clicked(
                    mid
                ),
            ),
            ft.TextButton(
                "Branch",
                on_click=lambda e, mid=message.message_id: self._on_branch_from_message_clicked(
                    mid
                ),
            ),
        ]
        if message.role == MessageRole.ASSISTANT.value:
            action_controls.insert(
                0,
                ft.TextButton(
                    "Copy",
                    on_click=lambda e, text=message.content: self._copy_text(text),
                ),
            )
            if self._is_last_assistant_message(message.message_id):
                action_controls.insert(
                    1,
                    ft.TextButton("Regenerate", on_click=self._on_regenerate_clicked),
                )
        return ft.Row(
            expand=True,
            alignment=(
                ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
            ),
            controls=[
                ft.Container(
                    width=340 if is_user else 680,
                    margin=ft.Margin.only(left=80) if not is_user else None,
                    padding=16,
                    border_radius=18,
                    bgcolor=background,
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.15, "#E2E8F0")),
                    content=ft.Column(
                        spacing=6,
                        controls=[
                            ft.Text(
                                label,
                                size=12,
                                color="#F59E0B",
                                weight=ft.FontWeight.W_600,
                            ),
                            content_control,
                            ft.Text(
                                self._build_message_meta_text(message),
                                size=11,
                                color="#94A3B8",
                            ),
                            ft.Row(
                                wrap=True,
                                spacing=6,
                                alignment=(
                                    ft.MainAxisAlignment.END
                                    if is_user
                                    else ft.MainAxisAlignment.START
                                ),
                                controls=action_controls,
                            ),
                        ],
                    ),
                )
            ],
        )

    def _build_chat_stats_text(self, status: ModelStatus) -> str:
        if status.error_message:
            return status.error_message
        return f"{status.model_state} / {status.generation_state}"

    def _build_chat_meta_text(self) -> tuple[str, str]:
        last_assistant = self._last_assistant_message()
        if last_assistant is None:
            return ("Ctrl+Enter sends. Shift+Enter inserts a new line.", "")
        meta = dict(last_assistant.generation_meta)
        total_seconds = float(meta.get("total_seconds", 0.0) or 0.0)
        output_tokens = int(meta.get("output_tokens", 0) or 0)
        tokens_per_second = float(meta.get("tokens_per_second", 0.0) or 0.0)
        first_token_seconds = float(meta.get("first_token_seconds", 0.0) or 0.0)
        output_mode = str(
            meta.get("output_mode", self.service.current_session.output_mode)
        )
        json_valid = meta.get("json_valid")
        top_line = [
            f"Mode: {output_mode}",
            f"TTFT: {first_token_seconds:.2f}s",
            f"Total: {total_seconds:.2f}s",
        ]
        if json_valid is True:
            top_line.append("JSON: ok")
        elif json_valid is False:
            top_line.append("JSON: retry needed")
        bottom_line = [
            f"Tokens: {output_tokens}",
            f"Tok/s: {tokens_per_second:.2f}",
        ]
        return ("   |   ".join(top_line), "   |   ".join(bottom_line))

    def _build_runtime_info_text(self) -> str:
        describe_runtime = getattr(self.service.backend, "describe_torch_runtime", None)
        if callable(describe_runtime):
            summary = describe_runtime().replace("\n", " | ")
        else:
            summary = "Torch runtime details are unavailable."
        prefix = (
            "CUDA available"
            if self.service.status.gpu_available
            else "CUDA unavailable"
        )
        return f"{prefix} | {summary}"

    def _build_quantization_hint_text(self) -> tuple[str, str]:
        quant = self.quant_dropdown.value or Quantization.NONE.value
        is_bitsandbytes_available = getattr(
            self.service.backend, "is_bitsandbytes_available", None
        )
        bitsandbytes_available = (
            bool(is_bitsandbytes_available())
            if callable(is_bitsandbytes_available)
            else False
        )
        if quant == Quantization.NONE.value:
            return (
                "Quantization: none. float16 / bfloat16 alone do not require bitsandbytes.",
                "#94A3B8",
            )
        if bitsandbytes_available:
            return (f"Quantization: {quant}. bitsandbytes is installed.", "#94A3B8")
        return (
            f"Quantization: {quant}. bitsandbytes is not installed, so this selection cannot load. "
            "Use quant='none' for plain float16 / bfloat16.",
            "#FCA5A5",
        )

    def _load_options_from_session_model(self) -> LoadOptions:
        model = self.service.current_session.model
        return LoadOptions(
            model_id=model.model_id or DEFAULT_MODEL_IDS[0],
            quant=model.quant or Quantization.NONE.value,
            dtype=model.dtype or DType.BFLOAT16.value,
            device_map=model.device_map or DeviceMap.AUTO.value,
        )

    def _find_matching_prompt_key(self, prompt: str) -> str:
        normalized = prompt.strip()
        for name, preset_prompt in SYSTEM_PROMPT_PRESETS.items():
            if preset_prompt.strip() == normalized:
                return name
        for name, preset_prompt in self.service.settings.system_prompt_presets.items():
            if preset_prompt.strip() == normalized:
                return name
        return CUSTOM_PROMPT_KEY

    def _prompt_text_for_key(self, key: str) -> str:
        if key in SYSTEM_PROMPT_PRESETS:
            return SYSTEM_PROMPT_PRESETS[key]
        return self.service.settings.system_prompt_presets.get(key, "")

    def _current_load_options(self) -> LoadOptions:
        self.selected_load_options = LoadOptions(
            model_id=self.model_id_dropdown.value or "",
            quant=self.quant_dropdown.value or Quantization.NONE.value,
            dtype=self.dtype_dropdown.value or DType.BFLOAT16.value,
            device_map=self.device_map_dropdown.value or DeviceMap.AUTO.value,
            revision=(self.revision_field.value or "").strip(),
        )
        return self.selected_load_options

    def _current_generation_parameters(self) -> GenerationParameters | None:
        try:
            temperature = float((self.temperature_field.value or "0.7").strip())
            top_p = float((self.top_p_field.value or "0.95").strip())
            max_new_tokens = int((self.max_new_tokens_field.value or "256").strip())
            repetition_penalty = float(
                (self.repetition_penalty_field.value or "1.05").strip()
            )
            seed_text = (self.seed_field.value or "").strip()
            seed = int(seed_text) if seed_text else None
        except ValueError:
            self._show_message(
                "Generation controls contain an invalid numeric value.", error=True
            )
            return None
        return GenerationParameters(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

    def _build_benchmark_spec(self) -> BenchmarkSpec | None:
        prompt_name = self.benchmark_prompt_dropdown.value or next(
            iter(PROMPT_PRESETS.keys()), None
        )
        if not prompt_name:
            self._show_message("Select a benchmark prompt.", error=True)
            return None
        load_options: list[tuple[str, LoadOptions]] = []
        for preset in MODEL_PRESETS:
            checkbox = self.benchmark_model_checks[preset.name]
            if not checkbox.value:
                continue
            load_options.append(
                (
                    preset.name,
                    LoadOptions(
                        model_id=preset.model_id,
                        quant=preset.quant,
                        dtype=preset.dtype,
                        device_map=preset.device_map,
                    ),
                )
            )
        if not load_options:
            self._show_message(
                "Select at least one model preset to benchmark.", error=True
            )
            return None
        return BenchmarkSpec(
            prompt_name=prompt_name,
            prompt_text=PROMPT_PRESETS[prompt_name],
            load_options=load_options,
        )

    def _show_message(self, message: str, error: bool = False) -> None:
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor="#991B1B" if error else "#1E293B",
        )
        self.page.snack_bar.open = True

    def _build_message_content_control(self, message: ChatMessage) -> ft.Control:
        if message.role == MessageRole.ASSISTANT.value and not message.content.strip():
            return ft.Row(
                spacing=10,
                controls=[
                    ft.ProgressRing(width=14, height=14, stroke_width=2),
                    ft.Text("Generating...", selectable=True),
                ],
            )
        message_output_mode = str(
            message.generation_meta.get(
                "output_mode", self.service.current_session.output_mode
            )
        )
        if message.role == MessageRole.ASSISTANT.value:
            if message_output_mode == OutputMode.CODE_ONLY.value:
                return self._build_code_block_panel(message.content)
            code_segments = self._parse_code_segments(message.content)
            if code_segments:
                return self._build_rich_assistant_message_content(
                    message.content, code_segments, message_output_mode
                )
            if (
                hasattr(ft, "Markdown")
                and message_output_mode == OutputMode.MARKDOWN.value
            ):
                return ft.Markdown(value=message.content)
        return ft.Text(message.content, selectable=True)

    def _estimate_chat_scroll_overflow(self, messages: list[ChatMessage]) -> bool:
        estimated_units = 0
        for message in messages:
            content_length = len((message.content or "").strip())
            estimated_units += 1
            estimated_units += max(0, content_length // 240)
            if message.thinking_text:
                estimated_units += 1
        return estimated_units >= 6

    def _update_chat_resend_overlay_visibility(self) -> bool:
        visible = (
            self._chat_surface_hovered
            and self._chat_scroll_has_overflow
            and self._last_user_message() is not None
            and not self.resend_button.disabled
        )
        changed = self.chat_resend_overlay.visible != visible
        self.chat_resend_overlay.visible = visible
        return changed

    def _on_chat_messages_hover(self, e: ft.ControlEvent) -> None:
        self._chat_surface_hovered = str(e.data).lower() == "true"
        if self._update_chat_resend_overlay_visibility():
            self.chat_resend_overlay.update()

    def _on_chat_messages_scrolled(self, e: ft.OnScrollEvent) -> None:
        self._chat_scroll_has_overflow = e.max_scroll_extent > 1
        if self._update_chat_resend_overlay_visibility():
            self.chat_resend_overlay.update()

    def _build_rich_assistant_message_content(
        self,
        content: str,
        code_segments: list[tuple[int, int, str, str]],
        message_output_mode: str,
    ) -> ft.Control:
        controls: list[ft.Control] = []
        cursor = 0
        for start, end, language, code in code_segments:
            before = content[cursor:start]
            if before.strip():
                controls.append(
                    self._build_assistant_text_segment(before, message_output_mode)
                )
            controls.append(self._build_code_block_panel(code, language))
            cursor = end
        after = content[cursor:]
        if after.strip():
            controls.append(
                self._build_assistant_text_segment(after, message_output_mode)
            )
        return ft.Column(spacing=10, controls=controls)

    def _build_assistant_text_segment(
        self, text: str, message_output_mode: str
    ) -> ft.Control:
        if hasattr(ft, "Markdown") and message_output_mode == OutputMode.MARKDOWN.value:
            return ft.Markdown(value=text.strip())
        return ft.Text(text.strip(), selectable=True)

    def _build_code_block_panel(self, code_text: str, language: str = "") -> ft.Control:
        header_label = language or "code"
        return ft.Container(
            border_radius=14,
            bgcolor="#0B1220",
            border=ft.Border.all(1, "#334155"),
            padding=12,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text(header_label, size=11, color="#94A3B8"),
                    ft.Text(
                        code_text.strip("\n"),
                        selectable=True,
                        font_family="Consolas",
                        size=13,
                    ),
                    ft.Row(
                        alignment=ft.MainAxisAlignment.END,
                        controls=[
                            ft.TextButton(
                                "Code Copy",
                                on_click=lambda e, text=code_text.strip(
                                    "\n"
                                ): self._copy_text(text),
                            )
                        ],
                    ),
                ],
            ),
        )

    def _parse_code_segments(self, content: str) -> list[tuple[int, int, str, str]]:
        segments: list[tuple[int, int, str, str]] = []
        for match in re.finditer(r"```([\w.+-]+)?\n(.*?)```", content, flags=re.DOTALL):
            code = match.group(2).strip("\n")
            if not code:
                continue
            segments.append((match.start(), match.end(), match.group(1) or "", code))
        return segments

    def _build_message_meta_text(self, message: ChatMessage) -> str:
        meta = dict(message.generation_meta)
        parts = [self._format_message_timestamp(message.updated_at)]
        if meta.get("output_mode"):
            parts.append(str(meta["output_mode"]))
        if meta.get("total_seconds"):
            parts.append(f"{float(meta['total_seconds']):.2f}s total")
        if meta.get("tokens_per_second"):
            parts.append(f"{float(meta['tokens_per_second']):.2f} tok/s")
        if meta.get("output_tokens"):
            parts.append(f"{int(meta['output_tokens'])} tok")
        if meta.get("notice"):
            parts.append(str(meta["notice"]))
        if message.thinking_text:
            parts.append("thinking")
        return " | ".join(part for part in parts if part)

    def _format_message_timestamp(self, value: str) -> str:
        if not value.strip():
            return ""
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        now = (
            datetime.now(parsed.tzinfo) if parsed.tzinfo is not None else datetime.now()
        )
        if parsed.date() == now.date():
            return parsed.strftime("%H:%M:%S")
        return parsed.strftime("%Y-%m-%d %H:%M")

    def _build_session_option_label(self, session: object) -> str:
        pinned = "PIN | " if getattr(session, "pinned", False) else ""
        updated_at = str(getattr(session, "updated_at", ""))
        trimmed_updated = updated_at.replace("T", " ").replace("Z", "")
        title = str(getattr(session, "title", "Untitled Session"))
        return f"{pinned}{title} | {trimmed_updated}".strip(" |")

    def _last_assistant_message(self) -> ChatMessage | None:
        for message in reversed(self.service.current_session.messages):
            if message.role == MessageRole.ASSISTANT.value:
                return message
        return None

    def _last_user_message(self) -> ChatMessage | None:
        for message in reversed(self.service.current_session.messages):
            if message.role == MessageRole.USER.value:
                return message
        return None

    def _is_last_assistant_message(self, message_id: str) -> bool:
        last_message = self._last_assistant_message()
        return bool(last_message and last_message.message_id == message_id)

    def _pending_user_message_is_committed(self, messages: list[ChatMessage]) -> bool:
        if not self._pending_user_message or not messages:
            return False
        last_message = messages[-1]
        return (
            last_message.role == "user"
            and last_message.content == self._pending_user_message
        )

    def _refresh_from_background(self) -> None:
        self._refresh_all()
        self.page.update()

    def _on_new_chat_clicked(self, e: ft.ControlEvent) -> None:
        if self.service.status.generation_state in {"GENERATING", "STOPPING"}:
            return
        self.service.request_new_session()
        self.selected_load_options = self._load_options_from_session_model()
        self._chat_input_draft = ""
        self._pending_user_message = ""
        self._editing_message_id = ""
        self._editing_message_role = ""
        self.message_edit_field.value = ""
        self.chat_input.value = ""
        self.current_view_index = 0
        self.content_host.content = self.view_pages[0]
        self._refresh_all()
        self.page.update()

    def _activate_session(self, session_id: str) -> None:
        if session_id:
            if not self.service.request_load_session(session_id):
                self._show_message(
                    self.service.status.error_message or "Failed to load session.",
                    error=True,
                )
                return
        else:
            self.service.request_new_session()
        self.selected_load_options = self._load_options_from_session_model()
        self._chat_input_draft = ""
        self._pending_user_message = ""
        self._editing_message_id = ""
        self._editing_message_role = ""
        self.message_edit_field.value = ""
        self.chat_input.value = ""
        self._refresh_all()
        self.page.update()

    def _on_session_selected(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        self._activate_session(self.session_dropdown.value or "")

    def _on_generate_clicked(self, e: ft.ControlEvent) -> None:
        params = self._current_generation_parameters()
        if params is None:
            return
        user_text = self._chat_input_draft or self.chat_input.value or ""
        cleaned_user_text = user_text.strip()
        if not cleaned_user_text:
            self._show_message("User input is empty.", error=True)
            self.page.update()
            return
        self._pending_user_message = cleaned_user_text
        self._chat_generation_pending = True
        self._chat_input_draft = ""
        self.chat_input.value = ""
        self._refresh_all()
        self.page.update()

        def on_started() -> None:
            self._chat_generation_pending = False
            self._pending_user_message = ""
            self._refresh_from_background()

        def on_stream(progress) -> None:  # type: ignore[no-untyped-def]
            self._refresh_chat_controls()
            self._refresh_status_bar()
            self.page.schedule_update()

        def worker() -> None:
            result = self.service.request_generate(
                user_text=cleaned_user_text,
                system_prompt=self.system_prompt_field.value or "",
                parameters=params,
                on_started=on_started,
                stream_callback=on_stream,
                output_mode=self.output_mode_dropdown.value or OutputMode.NORMAL.value,
                enable_thinking=bool(self.thinking_checkbox.value),
            )
            if result.success:
                self._chat_input_draft = ""
                self.chat_input.value = ""
                if result.json_valid is False:
                    self._show_message(
                        result.error_message or "The reply was not valid JSON.",
                        error=True,
                    )
                elif result.error_message:
                    self._show_message(result.error_message)
                else:
                    self._show_message(
                        f"Done. total={result.total_seconds:.2f}s tokens={result.output_tokens} tok/s={result.tokens_per_second:.2f}"
                    )
            else:
                if self._pending_user_message == cleaned_user_text:
                    self._pending_user_message = ""
                    self._chat_input_draft = cleaned_user_text
                    self.chat_input.value = cleaned_user_text
                self._show_message(
                    result.error_message or "Generation failed.", error=True
                )
            self._chat_generation_pending = False
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_stop_clicked(self, e: ft.ControlEvent) -> None:
        self.service.request_stop_generation()
        self._refresh_all()
        self.page.update()

    def _on_session_title_changed(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        title = (
            str(e.data)
            if e is not None and e.data is not None
            else (self.session_title_field.value or "")
        )
        self.service.set_current_session_title(title)

    def _on_session_title_blur(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        self.service.request_save_session(self.session_title_field.value or "")
        self._refresh_chat_controls()
        self.page.update()

    def _on_session_search_changed(self, e: ft.ControlEvent) -> None:
        self._session_search_query = (
            str(e.data)
            if e is not None and e.data is not None
            else (self.session_search_field.value or "")
        )
        self._refresh_chat_controls()
        self.page.update()

    def _on_session_filter_changed(self, e: ft.ControlEvent) -> None:
        self._session_filter_mode = self.session_filter_dropdown.value or "all"
        self._refresh_chat_controls()
        self.page.update()

    def _on_session_sort_changed(self, e: ft.ControlEvent) -> None:
        self._session_sort_mode = self.session_sort_dropdown.value or "updated_desc"
        self._refresh_chat_controls()
        self.page.update()

    def _on_pin_session_changed(self, e: ft.ControlEvent) -> None:
        self.service.set_current_session_pinned(bool(self.pin_session_checkbox.value))
        self._refresh_chat_controls()
        self.page.update()

    def _on_duplicate_session_clicked(self, e: ft.ControlEvent) -> None:
        duplicated = self.service.request_duplicate_session_by_id(
            self.service.current_session.session_id
        )
        if duplicated is None:
            self._show_message(
                self.service.status.error_message or "Failed to duplicate session.",
                error=True,
            )
            self.page.update()
            return
        self.selected_load_options = self._load_options_from_session_model()
        self._chat_input_draft = ""
        self._pending_user_message = ""
        self._editing_message_id = ""
        self._editing_message_role = ""
        self.message_edit_field.value = ""
        self.chat_input.value = ""
        self._refresh_all()
        self._show_message("Session duplicated.")
        self.page.update()

    def _on_session_duplicate_menu_clicked(self, session_id: str) -> None:
        duplicated = self.service.request_duplicate_session_by_id(session_id)
        if duplicated is None:
            self._show_message(
                self.service.status.error_message or "Failed to duplicate session.",
                error=True,
            )
            self.page.update()
            return
        self.selected_load_options = self._load_options_from_session_model()
        self._chat_input_draft = ""
        self._pending_user_message = ""
        self._editing_message_id = ""
        self._editing_message_role = ""
        self.message_edit_field.value = ""
        self.chat_input.value = ""
        self._refresh_all()
        self._show_message("Session duplicated.")
        self.page.update()

    def _on_session_pin_menu_clicked(self, session_id: str, is_pinned: bool) -> None:
        if not self.service.set_session_pinned(session_id, not is_pinned):
            self._show_message(
                self.service.status.error_message or "Failed to update pinned state.",
                error=True,
            )
            self.page.update()
            return
        self._refresh_all()
        self._show_message("Pinned updated.")
        self.page.update()

    def _on_session_delete_menu_clicked(self, session_id: str) -> None:
        deleting_current = session_id == self.service.current_session.session_id
        if not self.service.request_delete_session(session_id):
            self._show_message(
                self.service.status.error_message or "Failed to delete session.",
                error=True,
            )
            self.page.update()
            return
        if deleting_current:
            self.selected_load_options = self._load_options_from_session_model()
            self._chat_input_draft = ""
            self._pending_user_message = ""
            self._editing_message_id = ""
            self._editing_message_role = ""
            self.message_edit_field.value = ""
            self.chat_input.value = ""
        self._refresh_all()
        self._show_message("Session deleted.")
        self.page.update()

    def _on_session_export_markdown_menu_clicked(
        self, session_id: str, session_title: str
    ) -> None:
        if hasattr(self.page, "run_task"):
            self.page.run_task(
                self._export_session_markdown_for_id, session_id, session_title
            )
            return
        try:
            asyncio.create_task(
                self._export_session_markdown_for_id(session_id, session_title)
            )
        except RuntimeError:
            asyncio.run(self._export_session_markdown_for_id(session_id, session_title))

    def _on_session_export_json_menu_clicked(
        self, session_id: str, session_title: str
    ) -> None:
        if hasattr(self.page, "run_task"):
            self.page.run_task(
                self._export_session_json_for_id, session_id, session_title
            )
            return
        try:
            asyncio.create_task(
                self._export_session_json_for_id(session_id, session_title)
            )
        except RuntimeError:
            asyncio.run(self._export_session_json_for_id(session_id, session_title))

    def _on_chat_tool_tab_clicked(self, panel_key: str) -> None:
        if panel_key not in self.chat_tool_tabs:
            return
        self._active_chat_tool_panel = (
            "" if panel_key == self._active_chat_tool_panel else panel_key
        )
        self._refresh_chat_tool_panel()
        self.chat_page_stack.update()

    def _on_chat_tool_dismiss_clicked(self, e: ft.ControlEvent | None = None) -> None:
        if not self._active_chat_tool_panel:
            return
        self._active_chat_tool_panel = ""
        self._refresh_chat_tool_panel()
        self.chat_page_stack.update()

    def _on_output_mode_changed(self, e: ft.ControlEvent) -> None:
        self.service.set_current_output_mode(
            self.output_mode_dropdown.value or OutputMode.NORMAL.value
        )
        self._refresh_chat_controls()
        self.page.update()

    def _on_thinking_changed(self, e: ft.ControlEvent) -> None:
        self.service.set_current_thinking_enabled(bool(self.thinking_checkbox.value))
        self._refresh_chat_controls()
        self.page.update()

    def _on_regenerate_clicked(self, e: ft.ControlEvent | None) -> None:
        params = self._current_generation_parameters()
        if params is None:
            return
        self._chat_generation_pending = True
        self._refresh_all()
        self.page.update()

        def on_started() -> None:
            self._chat_generation_pending = False
            self._refresh_from_background()

        def on_stream(progress) -> None:  # type: ignore[no-untyped-def]
            self._refresh_chat_controls()
            self._refresh_status_bar()
            self.page.schedule_update()

        def worker() -> None:
            result = self.service.request_regenerate_last_response(
                parameters=params,
                on_started=on_started,
                stream_callback=on_stream,
            )
            if result.success:
                message = "Response regenerated."
                if result.json_valid is False:
                    self._show_message(
                        result.error_message
                        or "The regenerated reply was not valid JSON.",
                        error=True,
                    )
                else:
                    self._show_message(message)
            else:
                self._show_message(
                    result.error_message or "Regeneration failed.", error=True
                )
            self._chat_generation_pending = False
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_resend_clicked(self, e: ft.ControlEvent | None) -> None:
        params = self._current_generation_parameters()
        if params is None:
            return
        self._chat_generation_pending = True
        self._refresh_all()
        self.page.update()

        def on_started() -> None:
            self._chat_generation_pending = False
            self._refresh_from_background()

        def on_stream(progress) -> None:  # type: ignore[no-untyped-def]
            self._refresh_chat_controls()
            self._refresh_status_bar()
            self.page.schedule_update()

        def worker() -> None:
            result = self.service.request_resend_last_user_message(
                parameters=params,
                on_started=on_started,
                stream_callback=on_stream,
            )
            if result.success:
                self._show_message("Last user message sent again.")
            else:
                self._show_message(
                    result.error_message or "Failed to resend the last user message.",
                    error=True,
                )
            self._chat_generation_pending = False
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_edit_message_clicked(self, message_id: str) -> None:
        for message in self.service.current_session.messages:
            if message.message_id != message_id:
                continue
            self._editing_message_id = message_id
            self._editing_message_role = message.role
            self.message_edit_field.value = message.content
            break
        self._refresh_chat_controls()
        self.page.update()

    def _on_cancel_message_edit_clicked(self, e: ft.ControlEvent | None) -> None:
        self._editing_message_id = ""
        self._editing_message_role = ""
        self.message_edit_field.value = ""
        self._refresh_chat_controls()
        self.page.update()

    def _on_apply_message_edit_clicked(self, e: ft.ControlEvent | None) -> None:
        if not self._editing_message_id:
            return
        params = self._current_generation_parameters()
        if params is None:
            return

        def on_stream(progress) -> None:  # type: ignore[no-untyped-def]
            self._refresh_chat_controls()
            self._refresh_status_bar()
            self.page.schedule_update()

        def worker() -> None:
            result = self.service.request_edit_message(
                message_id=self._editing_message_id,
                new_text=self.message_edit_field.value or "",
                parameters=params,
                on_started=self._refresh_from_background,
                stream_callback=on_stream,
            )
            if result.success:
                self._editing_message_id = ""
                self._editing_message_role = ""
                self.message_edit_field.value = ""
                self._show_message("Message updated.")
            else:
                self._show_message(
                    result.error_message or "Message update failed.", error=True
                )
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_branch_from_message_clicked(self, message_id: str) -> None:
        self.service.request_branch_session_from_message(message_id)
        self.selected_load_options = self._load_options_from_session_model()
        self._editing_message_id = ""
        self._editing_message_role = ""
        self.message_edit_field.value = ""
        self._refresh_all()
        self._show_message("Created a branched session from this point.")
        self.page.update()

    async def _on_export_session_markdown_clicked(
        self, e: ft.ControlEvent | None
    ) -> None:
        await self._export_session_markdown_for_id(
            self.service.current_session.session_id,
            self.service.current_session.title or "session",
        )

    async def _export_session_markdown_for_id(
        self, session_id: str, session_title: str
    ) -> None:
        file_path = await ft.FilePicker().save_file(
            dialog_title="Export Session Markdown",
            file_name=f"{session_title or 'session'}.md",
            initial_directory=str(self.service.paths.data_dir),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["md"],
        )
        if not file_path:
            return
        try:
            output_path = self.service.export_session_markdown(
                session_id, Path(file_path)
            )
        except Exception as exc:
            self._show_message(str(exc), error=True)
            self.page.update()
            return
        self._show_message(f"Session exported: {output_path.name}")
        self._refresh_all()
        self.page.update()

    async def _on_export_session_json_clicked(self, e: ft.ControlEvent | None) -> None:
        await self._export_session_json_for_id(
            self.service.current_session.session_id,
            self.service.current_session.title or "session",
        )

    async def _export_session_json_for_id(
        self, session_id: str, session_title: str
    ) -> None:
        file_path = await ft.FilePicker().save_file(
            dialog_title="Export Session JSON",
            file_name=f"{session_title or 'session'}.json",
            initial_directory=str(self.service.paths.data_dir),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
        )
        if not file_path:
            return
        try:
            output_path = self.service.export_session_json(session_id, Path(file_path))
        except Exception as exc:
            self._show_message(str(exc), error=True)
            self.page.update()
            return
        self._show_message(f"Session exported: {output_path.name}")
        self._refresh_all()
        self.page.update()

    def _copy_text(self, text: str) -> None:
        if not text.strip():
            self._show_message("There is nothing to copy.", error=True)
            self.page.update()
            return
        if hasattr(self.page, "run_task"):
            self.page.run_task(self._copy_text_async, text)
            return
        set_clipboard = getattr(self.page, "set_clipboard", None)
        if callable(set_clipboard):
            set_clipboard(text)
        self._show_message("Copied to clipboard.")
        self.page.update()

    async def _copy_text_async(self, text: str) -> None:
        try:
            await self.clipboard_service.set(text)
        except Exception:
            set_clipboard = getattr(self.page, "set_clipboard", None)
            if callable(set_clipboard):
                set_clipboard(text)
            else:
                self._show_message(
                    "Clipboard copy is not available on this page.", error=True
                )
                self.page.update()
                return
        self._show_message("Copied to clipboard.")
        self.page.update()

    def _on_system_prompt_changed(self, e: ft.ControlEvent) -> None:
        if self._syncing or self._syncing_prompt:
            return
        prompt_text = (
            str(e.data)
            if e is not None and e.data is not None
            else (self.system_prompt_field.value or "")
        )
        self.service.set_current_system_prompt(prompt_text)

    def _on_system_prompt_blur(self, e: ft.ControlEvent) -> None:
        if self._syncing or self._syncing_prompt:
            return
        self._refresh_prompt_controls(
            sync_prompt_field=False, sync_preset_name_field=False
        )
        self.page.update()

    def _on_prompt_preset_changed(self, e: ft.ControlEvent) -> None:
        if self._syncing or self._syncing_prompt:
            return
        selected = self.prompt_preset_dropdown.value or CUSTOM_PROMPT_KEY
        if selected == CUSTOM_PROMPT_KEY:
            self._refresh_prompt_controls(sync_prompt_field=False)
            self.page.update()
            return
        prompt_text = self._prompt_text_for_key(selected)
        self._syncing_prompt = True
        self.system_prompt_field.value = prompt_text
        self.service.set_current_system_prompt(prompt_text)
        self.preset_name_field.value = (
            selected if selected in self.service.settings.system_prompt_presets else ""
        )
        self._syncing_prompt = False
        self._refresh_prompt_controls()
        self.page.update()

    def _on_preset_name_changed(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        self._preset_name_draft = (
            str(e.data)
            if e is not None and e.data is not None
            else (self.preset_name_field.value or "")
        )

    def _on_preset_name_blur(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        self._refresh_prompt_controls(
            sync_prompt_field=False, sync_preset_name_field=False
        )
        self.page.update()

    def _on_chat_input_focus(self, e: ft.ControlEvent) -> None:
        self._chat_input_has_focus = True

    def _on_chat_input_blur(self, e: ft.ControlEvent) -> None:
        self._chat_input_has_focus = False

    def _on_chat_input_changed(self, e: ft.ControlEvent) -> None:
        self._chat_input_draft = self.chat_input.value or ""

    def _on_page_keyboard_event(self, e: ft.KeyboardEvent) -> None:
        if not self._chat_input_has_focus or self.current_view_index != 0:
            return
        if not e.ctrl:
            return
        if (e.key or "").lower() not in {"enter", "numpad enter"}:
            return
        if self.chat_input.disabled:
            return
        if not (self._chat_input_draft or self.chat_input.value or "").strip():
            return
        if self._chat_shortcut_pending:
            return
        self._chat_shortcut_pending = True
        self.page.run_task(self._dispatch_generate_shortcut)

    async def _dispatch_generate_shortcut(self) -> None:
        try:
            await asyncio.sleep(0)
            self._on_generate_clicked(None)
        finally:
            self._chat_shortcut_pending = False

    def _on_create_prompt_preset(self, e: ft.ControlEvent) -> None:
        if self.service.save_system_prompt_preset(
            self._preset_name_draft or self.preset_name_field.value or "",
            self.service.current_session.system_prompt
            or self.system_prompt_field.value
            or "",
        ):
            self._show_message("Custom system prompt preset saved.")
        else:
            self._show_message(
                self.service.status.error_message or "Failed to save preset.",
                error=True,
            )
        self._refresh_all()
        self.page.update()

    def _on_delete_prompt_preset(self, e: ft.ControlEvent) -> None:
        selected_name = self.prompt_preset_dropdown.value or ""
        if self.service.delete_system_prompt_preset(selected_name):
            self._show_message("Custom system prompt preset deleted.")
        else:
            self._show_message(
                self.service.status.error_message or "Failed to delete preset.",
                error=True,
            )
        self._refresh_all()
        self.page.update()

    def _on_model_preset_changed(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        preset_name = self.model_preset_dropdown.value or ""
        preset = self.service.build_load_options_from_preset(preset_name)
        if preset is None:
            return
        self.selected_load_options = preset
        self._refresh_model_controls()
        self.page.update()

    def _on_model_options_changed(self, e: ft.ControlEvent) -> None:
        if self._syncing:
            return
        self._current_load_options()
        self._refresh_model_controls()
        self.page.update()

    def _on_load_model_clicked(self, e: ft.ControlEvent) -> None:
        options = self._current_load_options()
        normalized = TransformersBackend.normalize_model_id(options.model_id)
        if normalized != options.model_id:
            self.model_id_dropdown.value = normalized
            options = LoadOptions(
                model_id=normalized,
                quant=options.quant,
                dtype=options.dtype,
                device_map=options.device_map,
                revision=options.revision,
            )
            self.page.update()
        is_local = Path(options.model_id).expanduser().exists()
        is_preset = normalized in DEFAULT_MODEL_IDS
        if (
            not is_local
            and not is_preset
            and "/" in normalized
            and not options.revision
        ):
            self._show_message("Inspecting repository...")
            self.page.update()
            info = self.service.backend.inspect_repo(normalized)
            if info.get("error"):
                self._show_message(
                    f"Failed to inspect '{normalized}': {info['error']}", error=True
                )
                return
            if not info["has_transformers"]:
                if info["has_gguf"]:
                    self._show_message(
                        f"'{normalized}' contains only GGUF files. GGUF is not yet supported.",
                        error=True,
                    )
                else:
                    self._show_message(
                        f"'{normalized}' does not contain a Transformers model.",
                        error=True,
                    )
                return
            branches = info.get("branches", [])
            if len(branches) > 1:
                self._show_branch_selector(branches, options)
                return
        if options.quant != Quantization.NONE.value:
            is_bitsandbytes_available = getattr(
                self.service.backend, "is_bitsandbytes_available", None
            )
            if callable(is_bitsandbytes_available) and not is_bitsandbytes_available():
                build_message = getattr(
                    self.service.backend, "_build_missing_bitsandbytes_message", None
                )
                if callable(build_message):
                    message = build_message(options.quant)
                else:
                    message = "Selected quantization requires bitsandbytes. Set quantization to 'none' or install bitsandbytes."
                self._show_message(message, error=True)
                self._refresh_model_controls()
                self.page.update()
                return

        self._start_model_load(options)

    def _show_branch_selector(self, branches: list[str], options: LoadOptions) -> None:
        def on_select(e: ft.ControlEvent) -> None:
            dlg.open = False
            self.page.update()
            branch = e.control.data
            if branch and branch != "main":
                self.revision_field.value = branch
            load_options = LoadOptions(
                model_id=options.model_id,
                quant=options.quant,
                dtype=options.dtype,
                device_map=options.device_map,
                revision=branch or "",
            )
            self._start_model_load(load_options)

        dlg = ft.AlertDialog(
            title=ft.Text("Select Branch"),
            content=ft.Column(
                tight=True,
                controls=[
                    ft.Text(
                        f"'{options.model_id}' has multiple branches.\nSelect which branch to download:",
                        size=13,
                    ),
                    *[
                        ft.TextButton(branch, data=branch, on_click=on_select)
                        for branch in branches
                    ],
                ],
            ),
        )
        self.page.overlay.append(dlg)
        dlg.open = True
        self.page.update()

    def _start_model_load(self, options: LoadOptions) -> None:
        """Start model load in a background thread."""

        def worker() -> None:
            result = self.service.request_load_model(
                options, on_started=self._refresh_from_background
            )
            if result.success:
                source = self.service.backend.current_model_path or result.model_id
                self._show_message(f"Model ready locally: {source}")
            else:
                self._show_message(
                    result.error_message or "Model load failed.", error=True
                )
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_unload_model_clicked(self, e: ft.ControlEvent) -> None:
        def worker() -> None:
            success = self.service.request_unload_model(
                on_started=self._refresh_from_background
            )
            if success:
                self._show_message("Model unloaded.")
            else:
                self._show_message(
                    self.service.status.error_message or "Model unload failed.",
                    error=True,
                )
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_warmup_clicked(self, e: ft.ControlEvent) -> None:
        def worker() -> None:
            result = self.service.request_warmup(
                on_started=self._refresh_from_background
            )
            if result.success:
                self._show_message(
                    f"Warmup complete. total={result.total_seconds:.2f}s"
                )
            else:
                self._show_message(result.error_message or "Warmup failed.", error=True)
            self._refresh_from_background()

        self.page.run_thread(worker)

    def _on_run_benchmark_clicked(self, e: ft.ControlEvent) -> None:
        spec = self._build_benchmark_spec()
        if spec is None:
            self.page.update()
            return

        def progress(payload: dict[str, object]) -> None:
            step = payload.get("step", "?")
            total = payload.get("total", "?")
            phase = payload.get("phase", "")
            model_name = payload.get("model_name", "")
            self.benchmark_progress_text.value = f"{phase} {step}/{total}: {model_name}"
            self.page.schedule_update()

        def worker() -> None:
            self.benchmark_rows = self.service.request_run_benchmark(
                spec,
                on_started=self._refresh_from_background,
                progress_callback=progress,
            )
            if self.benchmark_rows:
                self.benchmark_progress_text.value = "Benchmark completed."
                self._show_message("Benchmark completed.")
            else:
                self._show_message(
                    self.service.status.error_message or "Benchmark failed.", error=True
                )
            self._refresh_from_background()

        self.page.run_thread(worker)

    async def _on_export_benchmark_clicked(self, e: ft.ControlEvent) -> None:
        if not self.benchmark_rows:
            self._show_message("There are no benchmark rows to export.", error=True)
            self.page.update()
            return
        default_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = await ft.FilePicker().save_file(
            dialog_title="Export Benchmark CSV",
            file_name=default_name,
            initial_directory=str(self.service.paths.data_dir),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv"],
        )
        if not file_path:
            return
        output_path = self.service.export_benchmark_csv(
            [row.to_dict() for row in self.benchmark_rows],
            Path(file_path),
        )
        self._show_message(f"Benchmark exported: {output_path}")
        self.page.update()

    def _on_save_settings_clicked(self, e: ft.ControlEvent) -> None:
        generation = self._current_generation_parameters()
        if generation is None:
            return
        settings = AppSettings(
            hf_cache_dir=(self.hf_cache_field.value or "").strip(),
            app_data_path=(self.app_data_field.value or "").strip(),
            theme=self.theme_dropdown.value or "dark",
            log_level=self.log_level_dropdown.value or "INFO",
            default_model_preset=self.default_model_dropdown.value or "Qwen3.5 2B BF16",
            default_generation=generation,
            system_prompt_presets=dict(self.service.settings.system_prompt_presets),
        )
        self.service.save_settings(settings)
        configure_logging(self.service.paths.log_path, settings.log_level)
        self.page.theme_mode = (
            ft.ThemeMode.DARK if settings.theme == "dark" else ft.ThemeMode.LIGHT
        )
        self.page.bgcolor = "#0F172A" if settings.theme == "dark" else "#F8FAFC"
        self._show_message("Settings saved.")
        self._refresh_all()
        self.page.update()


def _main(page: ft.Page) -> None:
    FletPlaygroundApp(page)


def main() -> None:
    paths = ensure_app_paths(build_app_paths())
    configure_logging(paths.log_path)
    ft.run(_main)
