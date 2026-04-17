use std::sync::Arc;

use agent::{CompactionConfig, Thread};
use agent_settings::AgentSettings;
use fs::Fs;
use gpui::{DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Window};
use language_model::LanguageModelRegistry;
use settings::{LanguageModelProviderSetting, LanguageModelSelection, Settings, update_settings_file};
use ui::{KeyBinding, Modal, ModalFooter, ModalHeader, prelude::*};
use ui_input::InputField;
use workspace::{ModalView, Workspace};

use crate::language_model_selector::language_model_selector;

fn resolve_configured_model(selection: &LanguageModelSelection, cx: &App) -> Option<language_model::ConfiguredModel> {
    let registry = LanguageModelRegistry::read_global(cx);
    let provider_id = language_model::LanguageModelProviderId(
        gpui::SharedString::from(selection.provider.0.clone()),
    );
    let provider = registry.provider(&provider_id)?;
    let model = provider
        .provided_models(cx)
        .iter()
        .find(|m| m.id().0 == selection.model.as_str())?
        .clone();
    Some(language_model::ConfiguredModel { provider, model })
}

fn make_model_picker(
    get_selection: impl Fn(&App) -> Option<LanguageModelSelection> + 'static,
    on_save: impl Fn(LanguageModelSelection, Arc<dyn Fs>, &App) + 'static,
    fs: Arc<dyn Fs>,
    focus_handle: FocusHandle,
    window: &mut Window,
    cx: &mut gpui::Context<CompactionSettingsModal>,
) -> Entity<crate::language_model_selector::LanguageModelSelector> {
    let fs_for_save = fs.clone();
    let fs_for_fav = fs.clone();
    cx.new(|cx| {
        language_model_selector(
            move |cx| {
                get_selection(cx).and_then(|selection| resolve_configured_model(&selection, cx))
            },
            {
                move |model, cx| {
                    let selection = LanguageModelSelection {
                        provider: LanguageModelProviderSetting(model.provider_id().0.to_string()),
                        model: model.id().0.to_string(),
                        enable_thinking: model.supports_thinking(),
                        effort: model.default_effort_level()
                            .map(|effort| effort.value.to_string()),
                        speed: None,
                    };
                    on_save(selection, fs_for_save.clone(), cx);
                }
            },
            {
                move |model, should_be_favorite, cx| {
                    crate::favorite_models::toggle_in_settings(
                        model, should_be_favorite, fs_for_fav.clone(), cx,
                    );
                }
            },
            false,
            focus_handle,
            window,
            cx,
        )
    })
}

pub struct CompactionSettingsModal {
    thread: Entity<Thread>,
    focus_handle: FocusHandle,
    auto_compact_enabled: bool,
    model_picker: Entity<crate::language_model_selector::LanguageModelSelector>,
    fast_model_picker: Entity<crate::language_model_selector::LanguageModelSelector>,
    standard_model_picker: Entity<crate::language_model_selector::LanguageModelSelector>,
    powerful_model_picker: Entity<crate::language_model_selector::LanguageModelSelector>,
    deep_omit_threshold_tokens: Entity<InputField>,
    summary_threshold_tokens: Entity<InputField>,
    min_content_chars: Entity<InputField>,
}

impl CompactionSettingsModal {
    pub fn toggle(
        thread: Entity<Thread>,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut gpui::Context<Workspace>,
    ) {
        workspace.toggle_modal(window, cx, |window, cx| Self::new(thread, window, cx));
    }

    fn new(thread: Entity<Thread>, window: &mut Window, cx: &mut gpui::Context<Self>) -> Self {
        let config = thread.read(cx).compaction_config().clone();
        let enabled = thread.read(cx).auto_compact_enabled();
        let focus_handle = cx.focus_handle();

        let fs = <dyn Fs>::global(cx);

        let model_picker = make_model_picker(
            |cx| AgentSettings::get_global(cx).thread_summary_model.clone(),
            |selection, fs, cx| {
                update_settings_file(fs, cx, move |settings, _cx| {
                    settings.agent.get_or_insert_default().thread_summary_model = Some(selection);
                });
            },
            fs.clone(),
            focus_handle.clone(),
            window,
            cx,
        );

        let fast_model_picker = make_model_picker(
            |cx| AgentSettings::get_global(cx).subagent_models.fast.clone(),
            |selection, fs, cx| {
                update_settings_file(fs, cx, move |settings, _cx| {
                    settings.agent.get_or_insert_default().subagent_models.get_or_insert_default().fast = Some(selection);
                });
            },
            fs.clone(),
            focus_handle.clone(),
            window,
            cx,
        );

        let standard_model_picker = make_model_picker(
            |cx| AgentSettings::get_global(cx).subagent_models.standard.clone(),
            |selection, fs, cx| {
                update_settings_file(fs, cx, move |settings, _cx| {
                    settings.agent.get_or_insert_default().subagent_models.get_or_insert_default().standard = Some(selection);
                });
            },
            fs.clone(),
            focus_handle.clone(),
            window,
            cx,
        );

        let powerful_model_picker = make_model_picker(
            |cx| AgentSettings::get_global(cx).subagent_models.powerful.clone(),
            |selection, fs, cx| {
                update_settings_file(fs, cx, move |settings, _cx| {
                    settings.agent.get_or_insert_default().subagent_models.get_or_insert_default().powerful = Some(selection);
                });
            },
            fs.clone(),
            focus_handle.clone(),
            window,
            cx,
        );

        let deep_omit_threshold_tokens = cx.new(|cx| {
            let input = InputField::new(window, cx, "80000")
                .label("Tool result strip boundary (tokens from end)")
                .tab_stop(true);
            input.set_text(&config.deep_omit_threshold_tokens.to_string(), window, cx);
            input
        });
        let summary_threshold_tokens = cx.new(|cx| {
            let input = InputField::new(window, cx, "80000")
                .label("Auto compact boundary (tokens from end)")
                .tab_stop(true);
            input.set_text(&config.summary_threshold_tokens.to_string(), window, cx);
            input
        });
        let min_content_chars = cx.new(|cx| {
            let input = InputField::new(window, cx, "60000")
                .label("Min content chars to summarize")
                .tab_stop(true);
            input.set_text(&config.min_content_chars.to_string(), window, cx);
            input
        });

        Self {
            thread,
            focus_handle,
            auto_compact_enabled: enabled,
            model_picker,
            fast_model_picker,
            standard_model_picker,
            powerful_model_picker,
            deep_omit_threshold_tokens,
            summary_threshold_tokens,
            min_content_chars,
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, _: &mut Window, cx: &mut gpui::Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn save_and_close(&mut self, _: &mut Window, cx: &mut gpui::Context<Self>) {
        self.save_config(cx);
        cx.emit(DismissEvent);
    }

    fn toggle_auto_compact(&mut self, _: &mut Window, cx: &mut gpui::Context<Self>) {
        self.auto_compact_enabled = !self.auto_compact_enabled;

        let fs = <dyn Fs>::global(cx);
        let enabled = self.auto_compact_enabled;
        update_settings_file(fs, cx, move |settings, _cx| {
            let agent = settings.agent.get_or_insert_default();
            let compaction = agent.compaction.get_or_insert_default();
            compaction.enabled = Some(enabled);
        });

        self.thread.update(cx, |thread, cx| {
            thread.set_auto_compact_enabled(self.auto_compact_enabled, cx);
        });
        cx.notify();
    }

    fn save_config(&self, cx: &mut gpui::Context<Self>) {
        let deep_omit_threshold_tokens = self
            .deep_omit_threshold_tokens
            .read(cx)
            .text(cx)
            .parse::<u64>()
            .unwrap_or(80_000);
        let summary_threshold_tokens = self
            .summary_threshold_tokens
            .read(cx)
            .text(cx)
            .parse::<u64>()
            .unwrap_or(80_000);
        let min_content_chars = self
            .min_content_chars
            .read(cx)
            .text(cx)
            .parse::<usize>()
            .unwrap_or(60_000);

        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _cx| {
            let agent = settings.agent.get_or_insert_default();
            let compaction = agent.compaction.get_or_insert_default();
            compaction.deep_omit_threshold_tokens = Some(deep_omit_threshold_tokens);
            compaction.summary_threshold_tokens = Some(summary_threshold_tokens);
            compaction.min_content_chars = Some(min_content_chars);
        });

        let config = CompactionConfig {
            deep_omit_threshold_tokens,
            summary_threshold_tokens,
            min_content_chars,
        };
        self.thread.update(cx, |thread, cx| {
            thread.set_compaction_config(config, cx);
        });
    }

    fn reset_defaults(&mut self, window: &mut Window, cx: &mut gpui::Context<Self>) {
        let default = CompactionConfig::default();
        self.deep_omit_threshold_tokens.update(cx, |input, cx| {
            input.set_text(&default.deep_omit_threshold_tokens.to_string(), window, cx);
        });
        self.summary_threshold_tokens.update(cx, |input, cx| {
            input.set_text(&default.summary_threshold_tokens.to_string(), window, cx);
        });
        self.min_content_chars.update(cx, |input, cx| {
            input.set_text(&default.min_content_chars.to_string(), window, cx);
        });

        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _cx| {
            if let Some(agent) = settings.agent.as_mut() {
                agent.compaction = None;
            }
        });

        self.thread.update(cx, |thread, cx| {
            thread.set_compaction_config(default, cx);
        });
        cx.notify();
    }
}

impl EventEmitter<DismissEvent> for CompactionSettingsModal {}

impl Focusable for CompactionSettingsModal {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl ModalView for CompactionSettingsModal {}

impl Render for CompactionSettingsModal {
    fn render(&mut self, window: &mut Window, cx: &mut gpui::Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle(cx);

        let window_size = window.viewport_size();
        let rem_size = window.rem_size();
        let modal_max_height = if window_size.height / rem_size > rems_from_px(600.).0 {
            rems_from_px(500.)
        } else {
            rems_from_px(250.)
        };

        v_flex()
            .id("compaction-settings-modal")
            .key_context("CompactionSettingsModal")
            .w(rems(36.))
            .elevation_3(cx)
            .on_action(cx.listener(Self::cancel))
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.focus_handle(cx).focus(window, cx);
            }))
            .child(
                Modal::new("compaction-settings", None)
                    .header(ModalHeader::new().headline("Compaction Settings"))
                    .child(
                        v_flex()
                            .id("compaction-settings-content")
                            .gap_3()
                            .py_2()
                            .max_h(modal_max_height)
                            .overflow_y_scroll()
                            .child(
                                v_flex().gap_1().child(
                                    div().text_sm().text_color(cx.theme().colors().text_muted).child("Summary Model")
                                ).child(self.model_picker.clone())
                            )
                            .child(div().h_px().bg(cx.theme().colors().border))
                            .child(
                                v_flex()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(cx.theme().colors().text_muted)
                                            .child("Subagent Models"),
                                    )
                                    .child(
                                        v_flex().gap_1().child(
                                            div().text_xs().text_color(cx.theme().colors().text_muted)
                                                .child("Fast — simple lookups, formatting, single edits")
                                        ).child(self.fast_model_picker.clone())
                                    )
                                    .child(
                                        v_flex().gap_1().child(
                                            div().text_xs().text_color(cx.theme().colors().text_muted)
                                                .child("Standard — code changes, tool calls, file reads")
                                        ).child(self.standard_model_picker.clone())
                                    )
                                    .child(
                                        v_flex().gap_1().child(
                                            div().text_xs().text_color(cx.theme().colors().text_muted)
                                                .child("Powerful — refactoring, architecture, deep analysis")
                                        ).child(self.powerful_model_picker.clone())
                                    )
                            )
                            .child(div().h_px().bg(cx.theme().colors().border))
                            .child(
                                h_flex()
                                    .justify_between()
                                    .items_center()
                                    .child(
                                        v_flex()
                                            .child(div().text_sm().child("Auto Compact"))
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(cx.theme().colors().text_muted)
                                                    .child(
                                                        "Automatically summarize old messages at the start of each turn",
                                                    ),
                                            ),
                                    )
                                    .child(
                                        Button::new(
                                            "auto-compact-toggle",
                                            if self.auto_compact_enabled {
                                                "On"
                                            } else {
                                                "Off"
                                            },
                                        )
                                        .style(if self.auto_compact_enabled {
                                            ButtonStyle::Filled
                                        } else {
                                            ButtonStyle::Outlined
                                        })
                                        .on_click(cx.listener(
                                            |this, _, window, cx| {
                                                this.toggle_auto_compact(window, cx);
                                            },
                                        )),
                                    ),
                            )
                            .child(div().h_px().bg(cx.theme().colors().border))
                            .child(
                                v_flex()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(cx.theme().colors().text_muted)
                                            .child("Tool Result Stripping (Tier 1)"),
                                    )
                                    .child(self.deep_omit_threshold_tokens.clone()),
                            )
                            .child(div().h_px().bg(cx.theme().colors().border))
                            .child(
                                v_flex()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(cx.theme().colors().text_muted)
                                            .child("Message Summary (Tier 2)"),
                                    )
                                    .child(self.summary_threshold_tokens.clone())
                                    .child(self.min_content_chars.clone()),
                            ),
                    )
                    .footer(
                        ModalFooter::new().end_slot(
                            h_flex()
                                .gap_1()
                                .child(
                                    Button::new("reset", "Reset to Defaults")
                                        .style(ButtonStyle::Outlined)
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.reset_defaults(window, cx);
                                        })),
                                )
                                .child(
                                    Button::new("cancel-btn", "Cancel")
                                        .key_binding(
                                            KeyBinding::for_action_in(
                                                &menu::Cancel,
                                                &focus_handle,
                                                cx,
                                            )
                                            .map(|kb| kb.size(rems_from_px(12.))),
                                        )
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.cancel(&menu::Cancel, window, cx);
                                        })),
                                )
                                .child(
                                    Button::new("save", "Save")
                                        .style(ButtonStyle::Filled)
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.save_and_close(window, cx);
                                        })),
                                ),
                        ),
                    ),
            )
    }
}