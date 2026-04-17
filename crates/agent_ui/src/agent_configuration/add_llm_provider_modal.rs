use std::sync::Arc;

use anyhow::Result;
use collections::HashSet;
use fs::Fs;
use gpui::{
    DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Render, ScrollHandle, Task,
};
use language_model::{LanguageModelProviderId, LanguageModelRegistry};
use language_models::provider::open_ai_compatible::{AvailableModel, ModelCapabilities};
use language_models::AllLanguageModelSettings;
use settings::{OpenAiCompatibleSettingsContent, Settings, update_settings_file};
use ui::{
    Banner, Checkbox, KeyBinding, Modal, ModalFooter, ModalHeader, Section, ToggleState,
    WithScrollbar, prelude::*,
};
use ui_input::InputField;
use workspace::{ModalView, Workspace};

fn single_line_input(
    label: impl Into<SharedString>,
    placeholder: &str,
    text: Option<&str>,
    tab_index: isize,
    window: &mut Window,
    cx: &mut App,
) -> Entity<InputField> {
    cx.new(|cx| {
        let input = InputField::new(window, cx, placeholder)
            .label(label)
            .tab_index(tab_index)
            .tab_stop(true);

        if let Some(text) = text {
            input.set_text(text, window, cx);
        }
        input
    })
}

#[derive(Clone, Copy)]
pub enum LlmCompatibleProvider {
    OpenAi,
}

impl LlmCompatibleProvider {
    fn name(&self) -> &'static str {
        match self {
            LlmCompatibleProvider::OpenAi => "OpenAI",
        }
    }

    fn api_url(&self) -> &'static str {
        match self {
            LlmCompatibleProvider::OpenAi => "https://api.openai.com/v1",
        }
    }
}

enum Mode {
    Add,
    Edit { provider_id: Arc<str> },
}

struct AddLlmProviderInput {
    provider_name: Entity<InputField>,
    api_url: Entity<InputField>,
    api_key: Entity<InputField>,
    models: Vec<ModelInput>,
}

impl AddLlmProviderInput {
    fn new(provider: LlmCompatibleProvider, window: &mut Window, cx: &mut App) -> Self {
        let provider_name =
            single_line_input("Provider Name", provider.name(), None, 1, window, cx);
        let api_url = single_line_input("API URL", provider.api_url(), None, 2, window, cx);
        let api_key = cx.new(|cx| {
            InputField::new(
                window,
                cx,
                "000000000000000000000000000000000000000000000000",
            )
            .label("API Key")
            .tab_index(3)
            .tab_stop(true)
            .masked(true)
        });

        Self {
            provider_name,
            api_url,
            api_key,
            models: vec![ModelInput::new(0, window, cx)],
        }
    }

    fn add_model(&mut self, window: &mut Window, cx: &mut App) {
        let model_index = self.models.len();
        self.models.push(ModelInput::new(model_index, window, cx));
    }

    fn remove_model(&mut self, index: usize) {
        self.models.remove(index);
    }

    fn new_for_edit(
        provider_id: &str,
        api_url: &str,
        models: &[AvailableModel],
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        let provider_name = single_line_input(
            "Provider Name",
            "e.g. my-openai-provider",
            Some(provider_id),
            1,
            window,
            cx,
        );
        let api_url_input = single_line_input(
            "API URL",
            "https://api.example.com/v1",
            Some(api_url),
            2,
            window,
            cx,
        );
        let api_key = cx.new(|cx| {
            InputField::new(window, cx, "Leave blank to keep current key")
                .label("API Key (optional)")
                .tab_index(3)
                .tab_stop(true)
                .masked(true)
        });

        let model_inputs: Vec<ModelInput> = if models.is_empty() {
            vec![ModelInput::new(0, window, cx)]
        } else {
            models
                .iter()
                .enumerate()
                .map(|(i, model)| ModelInput::from_existing(i, model, window, cx))
                .collect()
        };

        Self {
            provider_name,
            api_url: api_url_input,
            api_key,
            models: model_inputs,
        }
    }
}

struct ModelCapabilityToggles {
    pub supports_tools: ToggleState,
    pub supports_images: ToggleState,
    pub supports_parallel_tool_calls: ToggleState,
    pub supports_prompt_cache_key: ToggleState,
    pub supports_chat_completions: ToggleState,
}

struct ModelInput {
    name: Entity<InputField>,
    max_completion_tokens: Entity<InputField>,
    max_output_tokens: Entity<InputField>,
    max_tokens: Entity<InputField>,
    capabilities: ModelCapabilityToggles,
}

impl ModelInput {
    fn new(model_index: usize, window: &mut Window, cx: &mut App) -> Self {
        let base_tab_index = (3 + (model_index * 4)) as isize;

        let model_name = single_line_input(
            "Model Name",
            "e.g. gpt-5, claude-opus-4, gemini-2.5-pro",
            None,
            base_tab_index + 1,
            window,
            cx,
        );
        let max_completion_tokens = single_line_input(
            "Max Completion Tokens",
            "200000",
            Some("200000"),
            base_tab_index + 2,
            window,
            cx,
        );
        let max_output_tokens = single_line_input(
            "Max Output Tokens",
            "Max Output Tokens",
            Some("32000"),
            base_tab_index + 3,
            window,
            cx,
        );
        let max_tokens = single_line_input(
            "Max Tokens",
            "Max Tokens",
            Some("200000"),
            base_tab_index + 4,
            window,
            cx,
        );

        let ModelCapabilities {
            tools,
            images,
            parallel_tool_calls,
            prompt_cache_key,
            chat_completions,
            ..
        } = ModelCapabilities::default();

        Self {
            name: model_name,
            max_completion_tokens,
            max_output_tokens,
            max_tokens,
            capabilities: ModelCapabilityToggles {
                supports_tools: tools.into(),
                supports_images: images.into(),
                supports_parallel_tool_calls: parallel_tool_calls.into(),
                supports_prompt_cache_key: prompt_cache_key.into(),
                supports_chat_completions: chat_completions.into(),
            },
        }
    }

    fn from_existing(
        model_index: usize,
        model: &AvailableModel,
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        let base_tab_index = (3 + (model_index * 4)) as isize;

        let name = single_line_input(
            "Model Name",
            "e.g. gpt-5, claude-opus-4, gemini-2.5-pro",
            Some(&model.name),
            base_tab_index + 1,
            window,
            cx,
        );
        let max_completion_tokens_value = model.max_completion_tokens.map(|v| v.to_string());
        let max_completion_tokens = single_line_input(
            "Max Completion Tokens",
            "200000",
            max_completion_tokens_value.as_deref(),
            base_tab_index + 2,
            window,
            cx,
        );
        let max_output_tokens_value = model.max_output_tokens.map(|v| v.to_string());
        let max_output_tokens = single_line_input(
            "Max Output Tokens",
            "Max Output Tokens",
            max_output_tokens_value.as_deref(),
            base_tab_index + 3,
            window,
            cx,
        );
        let max_tokens_value = model.max_tokens.to_string();
        let max_tokens = single_line_input(
            "Max Tokens",
            "Max Tokens",
            Some(&max_tokens_value),
            base_tab_index + 4,
            window,
            cx,
        );

        Self {
            name,
            max_completion_tokens,
            max_output_tokens,
            max_tokens,
            capabilities: ModelCapabilityToggles {
                supports_tools: model.capabilities.tools.into(),
                supports_images: model.capabilities.images.into(),
                supports_parallel_tool_calls: model.capabilities.parallel_tool_calls.into(),
                supports_prompt_cache_key: model.capabilities.prompt_cache_key.into(),
                supports_chat_completions: model.capabilities.chat_completions.into(),
            },
        }
    }

    fn parse(&self, cx: &App) -> Result<AvailableModel, SharedString> {
        let name = self.name.read(cx).text(cx);
        if name.is_empty() {
            return Err(SharedString::from("Model Name cannot be empty"));
        }
        Ok(AvailableModel {
            name,
            display_name: None,
            max_completion_tokens: Some(
                self.max_completion_tokens
                    .read(cx)
                    .text(cx)
                    .parse::<u64>()
                    .map_err(|_| SharedString::from("Max Completion Tokens must be a number"))?,
            ),
            max_output_tokens: Some(
                self.max_output_tokens
                    .read(cx)
                    .text(cx)
                    .parse::<u64>()
                    .map_err(|_| SharedString::from("Max Output Tokens must be a number"))?,
            ),
            max_tokens: self
                .max_tokens
                .read(cx)
                .text(cx)
                .parse::<u64>()
                .map_err(|_| SharedString::from("Max Tokens must be a number"))?,
            reasoning_effort: None,
            capabilities: ModelCapabilities {
                tools: self.capabilities.supports_tools.selected(),
                images: self.capabilities.supports_images.selected(),
                parallel_tool_calls: self.capabilities.supports_parallel_tool_calls.selected(),
                prompt_cache_key: self.capabilities.supports_prompt_cache_key.selected(),
                chat_completions: self.capabilities.supports_chat_completions.selected(),
                interleaved_reasoning: false,
            },
        })
    }
}

fn save_provider_to_settings(
    input: &AddLlmProviderInput,
    mode: &Mode,
    cx: &mut App,
) -> Task<Result<(), SharedString>> {
    let provider_name: Arc<str> = input.provider_name.read(cx).text(cx).into();
    if provider_name.is_empty() {
        return Task::ready(Err("Provider Name cannot be empty".into()));
    }

    let api_url = input.api_url.read(cx).text(cx);
    if api_url.is_empty() {
        return Task::ready(Err("API URL cannot be empty".into()));
    }

    let api_key = input.api_key.read(cx).text(cx);
    if matches!(mode, Mode::Add) && api_key.is_empty() {
        return Task::ready(Err("API Key cannot be empty".into()));
    }

    let mut models = Vec::new();
    let mut model_names: HashSet<String> = HashSet::default();
    for model in &input.models {
        match model.parse(cx) {
            Ok(model) => {
                if !model_names.insert(model.name.clone()) {
                    return Task::ready(Err("Model Names must be unique".into()));
                }
                models.push(model)
            }
            Err(err) => return Task::ready(Err(err)),
        }
    }

    let fs = <dyn Fs>::global(cx);

    match mode {
        Mode::Add => save_new_provider(provider_name, api_url, api_key, models, fs, cx),
        Mode::Edit { provider_id } => update_existing_provider(
            provider_id.clone(),
            provider_name,
            api_url,
            api_key,
            models,
            fs,
            cx,
        ),
    }
}

fn save_new_provider(
    provider_name: Arc<str>,
    api_url: String,
    api_key: String,
    models: Vec<AvailableModel>,
    fs: Arc<dyn Fs>,
    cx: &mut App,
) -> Task<Result<(), SharedString>> {
    if LanguageModelRegistry::read_global(cx)
        .providers()
        .iter()
        .any(|provider| {
            provider.id().0.as_ref() == provider_name.as_ref()
                || provider.name().0.as_ref() == provider_name.as_ref()
        })
    {
        return Task::ready(Err(
            "Provider Name is already taken by another provider".into()
        ));
    }

    let task = cx.write_credentials(&api_url, "Bearer", api_key.as_bytes());
    cx.spawn(async move |cx| {
        task.await
            .map_err(|_| SharedString::from("Failed to write API key to keychain"))?;
        cx.update(|cx| {
            update_settings_file(fs, cx, |settings, _cx| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .openai_compatible
                    .get_or_insert_default()
                    .insert(
                        provider_name,
                        OpenAiCompatibleSettingsContent {
                            api_url,
                            available_models: models,
                        },
                    );
            });
        });
        Ok(())
    })
}

fn update_existing_provider(
    provider_id: Arc<str>,
    provider_name: Arc<str>,
    api_url: String,
    api_key: String,
    models: Vec<AvailableModel>,
    fs: Arc<dyn Fs>,
    cx: &mut App,
) -> Task<Result<(), SharedString>> {
    let name_conflict = LanguageModelRegistry::read_global(cx)
        .providers()
        .iter()
        .any(|provider| {
            provider.id().0.as_ref() == provider_name.as_ref()
                || provider.name().0.as_ref() == provider_name.as_ref()
        });

    if name_conflict {
        let is_same_provider = provider_name.as_ref() == provider_id.as_ref();
        if !is_same_provider {
            return Task::ready(Err(
                "Provider Name is already taken by another provider".into()
            ));
        }
    }

    let old_api_url = AllLanguageModelSettings::get_global(cx)
        .openai_compatible
        .get(&*provider_id)
        .map(|s| s.api_url.clone());
    let write_credentials_task = if api_key.is_empty() {
        None
    } else {
        Some(cx.write_credentials(&api_url, "Bearer", api_key.as_bytes()))
    };
    cx.spawn(async move |cx| {
        if let Some(ref old_url) = old_api_url {
            if old_url != &api_url {
                let delete_task = cx.update(|cx| cx.delete_credentials(old_url));
                if let Err(err) = delete_task.await {
                    log::warn!("Failed to delete old credentials: {err:#}");
                }
            }
        }
        if let Some(task) = write_credentials_task {
            task.await
                .map_err(|_| SharedString::from("Failed to write API key to keychain"))?;
        }
        let provider_name_for_registry = provider_name.clone();
        cx.update(|cx| {
            let provider_id = provider_id.clone();
            update_settings_file(fs, cx, move |settings, _cx| {
                let map = settings
                    .language_models
                    .get_or_insert_default()
                    .openai_compatible
                    .get_or_insert_default();
                if provider_name.as_ref() != provider_id.as_ref() {
                    map.remove(&provider_id);
                }
                map.insert(
                    provider_name,
                    OpenAiCompatibleSettingsContent {
                        api_url,
                        available_models: models,
                    },
                );
            });
        });
        cx.update(|cx| {
            let provider_id = provider_id.clone();
            LanguageModelRegistry::global(cx).update(cx, move |registry, cx| {
                if provider_name_for_registry.as_ref() != provider_id.as_ref() {
                    registry.unregister_provider(
                        LanguageModelProviderId(SharedString::new(provider_id)),
                        cx,
                    );
                }
            });
        });
        Ok(())
    })
}

pub struct AddLlmProviderModal {
    mode: Mode,
    provider: LlmCompatibleProvider,
    input: AddLlmProviderInput,
    scroll_handle: ScrollHandle,
    focus_handle: FocusHandle,
    last_error: Option<SharedString>,
}

impl AddLlmProviderModal {
    pub fn toggle(
        provider: LlmCompatibleProvider,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        workspace.toggle_modal(window, cx, |window, cx| {
            Self::new(provider, Mode::Add, window, cx)
        });
    }

    pub fn toggle_edit(
        provider_id: Arc<str>,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        workspace.toggle_modal(window, cx, |window, cx| {
            Self::new(
                LlmCompatibleProvider::OpenAi,
                Mode::Edit { provider_id },
                window,
                cx,
            )
        });
    }

    fn new(
        provider: LlmCompatibleProvider,
        mode: Mode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (input, mode) = match mode {
            Mode::Add => (AddLlmProviderInput::new(provider, window, cx), Mode::Add),
            Mode::Edit { provider_id } => {
                let provider_settings = AllLanguageModelSettings::get_global(cx)
                    .openai_compatible
                    .get(&provider_id)
                    .map(|settings| (settings.api_url.clone(), settings.available_models.clone()));
                match provider_settings {
                    Some((api_url, available_models)) => (
                        AddLlmProviderInput::new_for_edit(
                            &provider_id,
                            &api_url,
                            &available_models,
                            window,
                            cx,
                        ),
                        Mode::Edit { provider_id },
                    ),
                    None => (AddLlmProviderInput::new(provider, window, cx), Mode::Add),
                }
            }
        };
        Self {
            input,
            mode,
            provider,
            last_error: None,
            focus_handle: cx.focus_handle(),
            scroll_handle: ScrollHandle::new(),
        }
    }

    fn confirm(&mut self, _: &menu::Confirm, _: &mut Window, cx: &mut Context<Self>) {
        let task = save_provider_to_settings(&self.input, &self.mode, cx);
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| match result {
                Ok(_) => {
                    cx.emit(DismissEvent);
                }
                Err(error) => {
                    this.last_error = Some(error);
                    cx.notify();
                }
            })
        })
        .detach_and_log_err(cx);
    }

    fn cancel(&mut self, _: &menu::Cancel, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn render_model_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .mt_1()
            .gap_2()
            .child(
                h_flex()
                    .justify_between()
                    .child(Label::new("Models").size(LabelSize::Small))
                    .child(
                        Button::new("add-model", "Add Model")
                            .start_icon(
                                Icon::new(IconName::Plus)
                                    .size(IconSize::XSmall)
                                    .color(Color::Muted),
                            )
                            .label_size(LabelSize::Small)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.input.add_model(window, cx);
                                cx.notify();
                            })),
                    ),
            )
            .children(
                self.input
                    .models
                    .iter()
                    .enumerate()
                    .map(|(ix, _)| self.render_model(ix, cx)),
            )
    }

    fn render_model(&self, ix: usize, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let has_more_than_one_model = self.input.models.len() > 1;
        let model = &self.input.models[ix];

        v_flex()
            .p_2()
            .gap_2()
            .rounded_sm()
            .border_1()
            .border_dashed()
            .border_color(cx.theme().colors().border.opacity(0.6))
            .bg(cx.theme().colors().element_active.opacity(0.15))
            .child(model.name.clone())
            .child(
                h_flex()
                    .gap_2()
                    .child(model.max_completion_tokens.clone())
                    .child(model.max_output_tokens.clone()),
            )
            .child(model.max_tokens.clone())
            .child(
                v_flex()
                    .gap_1()
                    .child(
                        Checkbox::new(("supports-tools", ix), model.capabilities.supports_tools)
                            .label("Supports tools")
                            .on_click(cx.listener(move |this, checked, _window, cx| {
                                this.input.models[ix].capabilities.supports_tools = *checked;
                                cx.notify();
                            })),
                    )
                    .child(
                        Checkbox::new(("supports-images", ix), model.capabilities.supports_images)
                            .label("Supports images")
                            .on_click(cx.listener(move |this, checked, _window, cx| {
                                this.input.models[ix].capabilities.supports_images = *checked;
                                cx.notify();
                            })),
                    )
                    .child(
                        Checkbox::new(
                            ("supports-parallel-tool-calls", ix),
                            model.capabilities.supports_parallel_tool_calls,
                        )
                        .label("Supports parallel_tool_calls")
                        .on_click(cx.listener(
                            move |this, checked, _window, cx| {
                                this.input.models[ix]
                                    .capabilities
                                    .supports_parallel_tool_calls = *checked;
                                cx.notify();
                            },
                        )),
                    )
                    .child(
                        Checkbox::new(
                            ("supports-prompt-cache-key", ix),
                            model.capabilities.supports_prompt_cache_key,
                        )
                        .label("Supports prompt_cache_key")
                        .on_click(cx.listener(
                            move |this, checked, _window, cx| {
                                this.input.models[ix].capabilities.supports_prompt_cache_key =
                                    *checked;
                                cx.notify();
                            },
                        )),
                    )
                    .child(
                        Checkbox::new(
                            ("supports-chat-completions", ix),
                            model.capabilities.supports_chat_completions,
                        )
                        .label("Supports /chat/completions")
                        .on_click(cx.listener(
                            move |this, checked, _window, cx| {
                                this.input.models[ix].capabilities.supports_chat_completions =
                                    *checked;
                                cx.notify();
                            },
                        )),
                    ),
            )
            .when(has_more_than_one_model, |this| {
                this.child(
                    Button::new(("remove-model", ix), "Remove Model")
                        .start_icon(
                            Icon::new(IconName::Trash)
                                .size(IconSize::XSmall)
                                .color(Color::Muted),
                        )
                        .label_size(LabelSize::Small)
                        .style(ButtonStyle::Outlined)
                        .full_width()
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.input.remove_model(ix);
                            cx.notify();
                        })),
                )
            })
    }

    fn on_tab(&mut self, _: &menu::SelectNext, window: &mut Window, cx: &mut Context<Self>) {
        window.focus_next(cx);
    }

    fn on_tab_prev(
        &mut self,
        _: &menu::SelectPrevious,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus_prev(cx);
    }
}

impl EventEmitter<DismissEvent> for AddLlmProviderModal {}

impl Focusable for AddLlmProviderModal {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl ModalView for AddLlmProviderModal {}

impl Render for AddLlmProviderModal {
    fn render(&mut self, window: &mut ui::Window, cx: &mut ui::Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle(cx);

        let window_size = window.viewport_size();
        let rem_size = window.rem_size();
        let is_large_window = window_size.height / rem_size > rems_from_px(600.).0;

        let modal_max_height = if is_large_window {
            rems_from_px(450.)
        } else {
            rems_from_px(200.)
        };

        v_flex()
            .id("add-llm-provider-modal")
            .key_context("AddLlmProviderModal")
            .w(rems(34.))
            .elevation_3(cx)
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(Self::on_tab))
            .on_action(cx.listener(Self::on_tab_prev))
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.focus_handle(cx).focus(window, cx);
            }))
            .child(
                Modal::new("configure-context-server", None)
                    .header(ModalHeader::new()
                        .headline(match &self.mode {
                            Mode::Add => "Add LLM Provider",
                            Mode::Edit { .. } => "Edit LLM Provider",
                        })
                        .description(
                            match self.provider {
                                LlmCompatibleProvider::OpenAi => {
                                    "This provider will use an OpenAI compatible API."
                                }
                            },
                        ))
                    .when_some(self.last_error.clone(), |this, error| {
                        this.section(
                            Section::new().child(
                                Banner::new()
                                    .severity(Severity::Warning)
                                    .child(div().text_xs().child(error)),
                            ),
                        )
                    })
                    .child(
                        div()
                            .size_full()
                            .vertical_scrollbar_for(&self.scroll_handle, window, cx)
                            .child(
                                v_flex()
                                    .id("modal_content")
                                    .size_full()
                                    .tab_group()
                                    .max_h(modal_max_height)
                                    .pl_3()
                                    .pr_4()
                                    .pb_2()
                                    .gap_2()
                                    .overflow_y_scroll()
                                    .track_scroll(&self.scroll_handle)
                                    .child(self.input.provider_name.clone())
                                    .child(self.input.api_url.clone())
                                    .child(self.input.api_key.clone())
                                    .child(self.render_model_section(cx)),
                            ),
                    )
                    .footer(
                        ModalFooter::new().end_slot(
                            h_flex()
                                .gap_1()
                                .child(
                                    Button::new("cancel", "Cancel")
                                        .key_binding(
                                            KeyBinding::for_action_in(
                                                &menu::Cancel,
                                                &focus_handle,
                                                cx,
                                            )
                                            .map(|kb| kb.size(rems_from_px(12.))),
                                        )
                                        .on_click(cx.listener(|this, _event, window, cx| {
                                            this.cancel(&menu::Cancel, window, cx)
                                        })),
                                )
                                .child(
                                    Button::new("save-server", "Save Provider")
                                        .key_binding(
                                            KeyBinding::for_action_in(
                                                &menu::Confirm,
                                                &focus_handle,
                                                cx,
                                            )
                                            .map(|kb| kb.size(rems_from_px(12.))),
                                        )
                                        .on_click(cx.listener(|this, _event, window, cx| {
                                            this.confirm(&menu::Confirm, window, cx)
                                        })),
                                ),
                        ),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use gpui::{TestAppContext, VisualTestContext};
    use language_model::{
        LanguageModelProviderId, LanguageModelProviderName,
        fake_provider::FakeLanguageModelProvider,
    };
    use project::Project;
    use settings::SettingsStore;
    use util::path;
    use workspace::MultiWorkspace;

    #[gpui::test]
    async fn test_save_provider_invalid_inputs(cx: &mut TestAppContext) {
        let cx = setup_test(cx).await;

        assert_eq!(
            save_provider_validation_errors("", "someurl", "somekey", vec![], cx,).await,
            Some("Provider Name cannot be empty".into())
        );

        assert_eq!(
            save_provider_validation_errors("someprovider", "", "somekey", vec![], cx,).await,
            Some("API URL cannot be empty".into())
        );

        assert_eq!(
            save_provider_validation_errors("someprovider", "someurl", "", vec![], cx,).await,
            Some("API Key cannot be empty".into())
        );

        assert_eq!(
            save_provider_validation_errors(
                "someprovider",
                "someurl",
                "somekey",
                vec![("", "200000", "200000", "32000")],
                cx,
            )
            .await,
            Some("Model Name cannot be empty".into())
        );

        assert_eq!(
            save_provider_validation_errors(
                "someprovider",
                "someurl",
                "somekey",
                vec![("somemodel", "abc", "200000", "32000")],
                cx,
            )
            .await,
            Some("Max Tokens must be a number".into())
        );

        assert_eq!(
            save_provider_validation_errors(
                "someprovider",
                "someurl",
                "somekey",
                vec![("somemodel", "200000", "abc", "32000")],
                cx,
            )
            .await,
            Some("Max Completion Tokens must be a number".into())
        );

        assert_eq!(
            save_provider_validation_errors(
                "someprovider",
                "someurl",
                "somekey",
                vec![("somemodel", "200000", "200000", "abc")],
                cx,
            )
            .await,
            Some("Max Output Tokens must be a number".into())
        );

        assert_eq!(
            save_provider_validation_errors(
                "someprovider",
                "someurl",
                "somekey",
                vec![
                    ("somemodel", "200000", "200000", "32000"),
                    ("somemodel", "200000", "200000", "32000"),
                ],
                cx,
            )
            .await,
            Some("Model Names must be unique".into())
        );
    }

    #[gpui::test]
    async fn test_save_provider_name_conflict(cx: &mut TestAppContext) {
        let cx = setup_test(cx).await;

        cx.update(|_window, cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(
                    Arc::new(FakeLanguageModelProvider::new(
                        LanguageModelProviderId::new("someprovider"),
                        LanguageModelProviderName::new("Some Provider"),
                    )),
                    cx,
                );
            });
        });

        assert_eq!(
            save_provider_validation_errors(
                "someprovider",
                "someurl",
                "someapikey",
                vec![("somemodel", "200000", "200000", "32000")],
                cx,
            )
            .await,
            Some("Provider Name is already taken by another provider".into())
        );
    }

    #[gpui::test]
    async fn test_model_input_default_capabilities(cx: &mut TestAppContext) {
        let cx = setup_test(cx).await;

        cx.update(|window, cx| {
            let model_input = ModelInput::new(0, window, cx);
            model_input.name.update(cx, |input, cx| {
                input.set_text("somemodel", window, cx);
            });
            assert_eq!(
                model_input.capabilities.supports_tools,
                ToggleState::Selected
            );
            assert_eq!(
                model_input.capabilities.supports_images,
                ToggleState::Unselected
            );
            assert_eq!(
                model_input.capabilities.supports_parallel_tool_calls,
                ToggleState::Unselected
            );
            assert_eq!(
                model_input.capabilities.supports_prompt_cache_key,
                ToggleState::Unselected
            );
            assert_eq!(
                model_input.capabilities.supports_chat_completions,
                ToggleState::Selected
            );

            let parsed_model = model_input.parse(cx).unwrap();
            assert!(parsed_model.capabilities.tools);
            assert!(!parsed_model.capabilities.images);
            assert!(!parsed_model.capabilities.parallel_tool_calls);
            assert!(!parsed_model.capabilities.prompt_cache_key);
            assert!(parsed_model.capabilities.chat_completions);
        });
    }

    #[gpui::test]
    async fn test_model_input_deselected_capabilities(cx: &mut TestAppContext) {
        let cx = setup_test(cx).await;

        cx.update(|window, cx| {
            let mut model_input = ModelInput::new(0, window, cx);
            model_input.name.update(cx, |input, cx| {
                input.set_text("somemodel", window, cx);
            });

            model_input.capabilities.supports_tools = ToggleState::Unselected;
            model_input.capabilities.supports_images = ToggleState::Unselected;
            model_input.capabilities.supports_parallel_tool_calls = ToggleState::Unselected;
            model_input.capabilities.supports_prompt_cache_key = ToggleState::Unselected;
            model_input.capabilities.supports_chat_completions = ToggleState::Unselected;

            let parsed_model = model_input.parse(cx).unwrap();
            assert!(!parsed_model.capabilities.tools);
            assert!(!parsed_model.capabilities.images);
            assert!(!parsed_model.capabilities.parallel_tool_calls);
            assert!(!parsed_model.capabilities.prompt_cache_key);
            assert!(!parsed_model.capabilities.chat_completions);
        });
    }

    #[gpui::test]
    async fn test_model_input_with_name_and_capabilities(cx: &mut TestAppContext) {
        let cx = setup_test(cx).await;

        cx.update(|window, cx| {
            let mut model_input = ModelInput::new(0, window, cx);
            model_input.name.update(cx, |input, cx| {
                input.set_text("somemodel", window, cx);
            });

            model_input.capabilities.supports_tools = ToggleState::Selected;
            model_input.capabilities.supports_images = ToggleState::Unselected;
            model_input.capabilities.supports_parallel_tool_calls = ToggleState::Selected;
            model_input.capabilities.supports_prompt_cache_key = ToggleState::Unselected;
            model_input.capabilities.supports_chat_completions = ToggleState::Selected;

            let parsed_model = model_input.parse(cx).unwrap();
            assert_eq!(parsed_model.name, "somemodel");
            assert!(parsed_model.capabilities.tools);
            assert!(!parsed_model.capabilities.images);
            assert!(parsed_model.capabilities.parallel_tool_calls);
            assert!(!parsed_model.capabilities.prompt_cache_key);
            assert!(parsed_model.capabilities.chat_completions);
        });
    }

    async fn setup_test(cx: &mut TestAppContext) -> &mut VisualTestContext {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);

            language_model::init(cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn Fs>::set_global(fs.clone(), cx));
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let _workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        cx
    }

    async fn save_provider_validation_errors(
        provider_name: &str,
        api_url: &str,
        api_key: &str,
        models: Vec<(&str, &str, &str, &str)>,
        cx: &mut VisualTestContext,
    ) -> Option<SharedString> {
        fn set_text(input: &Entity<InputField>, text: &str, window: &mut Window, cx: &mut App) {
            input.update(cx, |input, cx| {
                input.set_text(text, window, cx);
            });
        }

        let task = cx.update(|window, cx| {
            let mut input = AddLlmProviderInput::new(LlmCompatibleProvider::OpenAi, window, cx);
            set_text(&input.provider_name, provider_name, window, cx);
            set_text(&input.api_url, api_url, window, cx);
            set_text(&input.api_key, api_key, window, cx);

            for (i, (name, max_tokens, max_completion_tokens, max_output_tokens)) in
                models.iter().enumerate()
            {
                if i >= input.models.len() {
                    input.models.push(ModelInput::new(i, window, cx));
                }
                let model = &mut input.models[i];
                set_text(&model.name, name, window, cx);
                set_text(&model.max_tokens, max_tokens, window, cx);
                set_text(
                    &model.max_completion_tokens,
                    max_completion_tokens,
                    window,
                    cx,
                );
                set_text(&model.max_output_tokens, max_output_tokens, window, cx);
            }
            save_provider_to_settings(&input, &Mode::Add, cx)
        });

        task.await.err()
    }
}
