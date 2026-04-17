use agent::{MessageSummary, Thread};
use anyhow::Result;
use futures::StreamExt;
use gpui::{DismissEvent, DragMoveEvent, Entity, EventEmitter, FocusHandle, Focusable, ScrollHandle};
use util::ResultExt;
use language_model::LanguageModelCompletionEvent;
use ui::{Banner, KeyBinding, Modal, ModalFooter, ModalHeader, Section, prelude::*};
use ui_input::InputField;
use workspace::{ModalView, Workspace};

#[derive(Debug, Clone)]
struct SliderDragValue;

enum CompactPhase {
    Selecting,
    Generating,
    Reviewing { summary_text: String, is_editing: bool },
}


pub struct CompactMessagesModal {
    thread: Entity<Thread>,
    focus_handle: FocusHandle,
    scroll_handle: ScrollHandle,

    boundary: usize,
    max_boundary: usize,
    summaries: Vec<MessageSummary>,

    phase: CompactPhase,
    last_error: Option<SharedString>,
    summary_input: Option<Entity<InputField>>,
}

impl CompactMessagesModal {
    pub fn toggle(
        thread: Entity<Thread>,
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut gpui::Context<Workspace>,
    ) {
        workspace.toggle_modal(window, cx, |window, cx| Self::new(thread, window, cx));
    }

    fn new(thread: Entity<Thread>, _window: &mut Window, cx: &mut gpui::Context<Self>) -> Self {
        let summaries = thread.read(cx).message_summaries_for_compaction();
        let max_boundary = summaries.len();
        let boundary = if max_boundary > 4 {
            (max_boundary / 2).max(2)
        } else if max_boundary > 0 {
            max_boundary
        } else {
            0
        };

        Self {
            thread,
            focus_handle: cx.focus_handle(),
            scroll_handle: ScrollHandle::new(),
            boundary,
            max_boundary,
            summaries,
            phase: CompactPhase::Selecting,
            last_error: None,
            summary_input: None,
        }
    }

    fn set_boundary_from_ratio(&mut self, ratio: f32, cx: &mut gpui::Context<Self>) {
        let new_boundary = if self.max_boundary == 0 {
            0
        } else {
            ((ratio * self.max_boundary as f32).round() as usize).clamp(0, self.max_boundary)
        };
        if new_boundary != self.boundary {
            self.boundary = new_boundary;
            cx.notify();
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, _: &mut Window, cx: &mut gpui::Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn generate_summary(&mut self, _window: &mut Window, cx: &mut gpui::Context<Self>) {
        if self.boundary == 0 {
            self.last_error = Some("No messages selected to compact".into());
            cx.notify();
            return;
        }

        let boundary = self.boundary;
        let preparation = self.thread.update(cx, |thread, _cx| {
            thread.prepare_manual_compaction(boundary)
        });

        let preparation = match preparation {
            Ok(Some(prep)) => prep,
            Ok(None) => {
                self.last_error = Some(
                    "No summarization model configured. Please configure a model first.".into(),
                );
                cx.notify();
                return;
            }
            Err(err) => {
                self.last_error = Some(format!("Failed to prepare compaction: {err:#}").into());
                cx.notify();
                return;
            }
        };

        self.phase = CompactPhase::Generating;
        self.last_error = None;
        cx.notify();

        let (model, request) = preparation;
        cx.spawn(async move |this, cx| {
            let mut summary_text = String::new();
            let result: Result<()> = async {
                let mut stream = model.stream_completion(request, cx).await?;
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(LanguageModelCompletionEvent::Text(text)) => {
                            summary_text.push_str(&text);
                        }
                        Ok(LanguageModelCompletionEvent::Stop(_)) => break,
                        Ok(_) => {}
                        Err(err) => return Err(err.into()),
                    }
                }
                Ok(())
            }
            .await;

            this.update(cx, |this, cx| {
                match result {
                    Ok(()) if !summary_text.is_empty() => {
                        this.summary_input = None;
                        this.phase = CompactPhase::Reviewing { summary_text, is_editing: false };
                    }
                    Ok(()) => {
                        this.phase = CompactPhase::Selecting;
                        this.last_error = Some("LLM returned an empty summary".into());
                    }
                    Err(err) => {
                        this.phase = CompactPhase::Selecting;
                        this.last_error =
                            Some(format!("Failed to generate summary: {err:#}").into());
                    }
                }
                cx.notify();
            })
            .log_err();
        })
        .detach();
    }

    fn toggle_edit(&mut self, _: &mut Window, cx: &mut gpui::Context<Self>) {
        if let CompactPhase::Reviewing { summary_text, is_editing } = &mut self.phase {
            if *is_editing {
                if let Some(input) = self.summary_input.as_ref() {
                    *summary_text = input.read(cx).text(cx);
                }
                self.summary_input = None;
                *is_editing = false;
            } else {
                self.summary_input = None;
                *is_editing = true;
            }
            cx.notify();
        }
    }

    fn replace_messages(&mut self, _window: &mut Window, cx: &mut gpui::Context<Self>) {
        let boundary = self.boundary;
        let summary_text = if let CompactPhase::Reviewing { summary_text, is_editing } = &self.phase {
            if *is_editing {
                self.summary_input.as_ref().map(|input| input.read(cx).text(cx)).unwrap_or_default()
            } else {
                summary_text.clone()
            }
        } else {
            return;
        };

        if summary_text.is_empty() {
            return;
        }

        self.thread.update(cx, |thread, cx| {
            thread.apply_manual_compaction(boundary, summary_text, cx);
        });

        cx.emit(DismissEvent);
    }

    fn back_to_selecting(&mut self, _window: &mut Window, cx: &mut gpui::Context<Self>) {
        self.phase = CompactPhase::Selecting;
        self.last_error = None;
        self.summary_input = None;
        cx.notify();
    }

    fn render_selecting_phase(&mut self, cx: &mut gpui::Context<Self>) -> impl IntoElement {
        let message_count = self.summaries.len();
        let compact_count = self.boundary.min(message_count);
        let keep_count = message_count.saturating_sub(compact_count);

        let selected_summaries: Vec<&MessageSummary> = self
            .summaries
            .iter()
            .filter(|s| s.index < self.boundary)
            .collect();

        let ratio = if self.max_boundary > 0 {
            self.boundary as f32 / self.max_boundary as f32
        } else {
            0.0
        };

        v_flex()
            .gap_3()
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!(
                        "Compact the oldest {compact_count} of {message_count} messages ({keep_count} will be kept)"
                    )),
            )
            .child(
                div()
                    .text_sm()
                    .child(format!("Messages to compact: {}", self.boundary)),
            )
            .child(
                div()
                    .id("compact-slider-track")
                    .w_full()
                    .h(px(28.))
                    .flex()
                    .items_center()
                    .cursor_ew_resize()
                    .on_drag(
                        SliderDragValue,
                        |_, _, _, cx| cx.new(|_| gpui::Empty),
                    )
                    .on_drag_move::<SliderDragValue>(
                        cx.listener(|this, event: &DragMoveEvent<SliderDragValue>, _, cx| {
                            let bounds = event.bounds;
                            let bounds_width = bounds.right() - bounds.left();
                            if bounds_width > px(0.) {
                                let ratio = ((event.event.position.x - bounds.left()) / bounds_width)
                                    .clamp(0.0, 1.0);
                                this.set_boundary_from_ratio(ratio, cx);
                            }
                        }),
                    )
                    .child(
                        div()
                            .w_full()
                            .h(px(6.))
                            .rounded_full()
                            .bg(cx.theme().colors().border)
                            .relative()
                            .child(
                                div()
                                    .absolute()
                                    .left_0()
                                    .top_0()
                                    .h_full()
                                    .rounded_full()
                                    .bg(cx.theme().colors().text_accent)
                                    .w(relative(if self.max_boundary > 0 {
                                        (self.boundary as f32 / self.max_boundary as f32).clamp(0.02, 1.0)
                                    } else {
                                        0.0
                                    })),
                            )
                            .child(
                                div()
                                    .absolute()
                                    .top(px(-5.))
                                    .left(relative(ratio))
                                    .ml(px(-7.))
                                    .size(px(16.))
                                    .rounded_full()
                                    .bg(cx.theme().colors().text_accent)
                                    .border_1()
                                    .border_color(cx.theme().colors().background),
                            ),
                    ),
            )
            .child(
                h_flex()
                    .justify_between()
                    .text_xs()
                    .text_color(cx.theme().colors().text_muted)
                    .child("0")
                    .child(format!("{}", self.max_boundary)),
            )
            .child(
                v_flex()
                    .id("message-preview-list")
                    .max_h(px(200.))
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .rounded_md()
                    .p_2()
                    .children(selected_summaries.iter().map(|summary| {
                        let role_label = if summary.is_user { "User" } else { "Agent" };
                        let role_color = if summary.is_user {
                            cx.theme().colors().text_accent
                        } else {
                            cx.theme().colors().text
                        };
                        div()
                            .flex()
                            .flex_row()
                            .gap_2()
                            .py_0p5()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(role_color)
                                    .w(px(48.))
                                    .child(role_label),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(cx.theme().colors().text_muted)
                                    .overflow_x_hidden()
                                    .child(summary.preview.clone()),
                            )
                    })),
            )
    }

    fn render_reviewing_phase(
        &self,
        cx: &mut gpui::Context<Self>,
    ) -> impl IntoElement {
        let (summary_text, is_editing) = match &self.phase {
            CompactPhase::Reviewing { summary_text, is_editing } => (summary_text.clone(), *is_editing),
            _ => (String::new(), false),
        };

        let summary_input = self.summary_input.clone();

        v_flex()
            .gap_3()
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!(
                        "This will replace {} messages with the summary below:",
                        self.boundary
                    )),
            )
            .child(
                if is_editing {
                    v_flex()
                        .id("summary-edit-scroll")
                        .max_h(px(300.))
                        .overflow_y_scroll()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .rounded_md()
                        .p_2()
                        .when_some(summary_input, |container, input| {
                            container.child(input)
                        })
                        .into_any_element()
                } else {
                    v_flex()
                        .id("summary-review-scroll")
                        .max_h(px(300.))
                        .overflow_y_scroll()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .rounded_md()
                        .p_3()
                        .text_sm()
                        .child(summary_text)
                        .into_any_element()
                }
            )
    }
}

impl EventEmitter<DismissEvent> for CompactMessagesModal {}

impl Focusable for CompactMessagesModal {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl ModalView for CompactMessagesModal {}

impl Render for CompactMessagesModal {
    fn render(&mut self, window: &mut Window, cx: &mut gpui::Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle(cx);
        let is_selecting = matches!(self.phase, CompactPhase::Selecting);
        let is_generating = matches!(self.phase, CompactPhase::Generating);
        let is_editing = matches!(&self.phase, CompactPhase::Reviewing { is_editing: true, .. });

        let headline = if is_selecting || is_generating {
            "Compact Old Messages"
        } else {
            "Review Summary"
        };

        if is_editing && self.summary_input.is_none() {
            if let CompactPhase::Reviewing { summary_text, .. } = &self.phase {
                let text = summary_text.clone();
                let input = cx.new(|cx| {
                    let input = InputField::new(window, cx, "").tab_stop(true);
                    input.set_text(&text, window, cx);
                    input
                });
                self.summary_input = Some(input);
            }
        }

        let content = if is_selecting {
            self.render_selecting_phase(cx).into_any_element()
        } else if is_generating {
            v_flex()
                .items_center()
                .justify_center()
                .py_8()
                .child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().colors().text_muted)
                        .child("Generating summary..."),
                )
                .into_any_element()
        } else {
            self.render_reviewing_phase(cx)
                .into_any_element()
        };

        let footer = if is_selecting {
            ModalFooter::new().end_slot(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new("cancel", "Cancel")
                            .key_binding(
                                KeyBinding::for_action_in(&menu::Cancel, &focus_handle, cx)
                                    .map(|kb| kb.size(rems_from_px(12.))),
                            )
                            .on_click(cx.listener(|this, _event, window, cx| {
                                this.cancel(&menu::Cancel, window, cx)
                            })),
                    )
                    .child(
                        Button::new("generate", "Generate Summary")
                            .style(ButtonStyle::Filled)
                            .disabled(self.boundary == 0 || self.max_boundary == 0)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.generate_summary(window, cx);
                            })),
                    ),
            )
        } else if is_generating {
            ModalFooter::new().end_slot(
                h_flex().gap_1().child(
                    Button::new("cancel-generating", "Cancel").on_click(cx.listener(
                        |this, _event, window, cx| {
                            this.cancel(&menu::Cancel, window, cx);
                        },
                    )),
                ),
            )
        } else {
            let edit_label = if is_editing { "Done Editing" } else { "Edit" };
            ModalFooter::new().end_slot(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new("edit-toggle", edit_label)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.toggle_edit(window, cx);
                            })),
                    )
                    .child(
                        Button::new("back", "Back").on_click(cx.listener(
                            |this, _, window, cx| {
                                this.back_to_selecting(window, cx);
                            },
                        )),
                    )
                    .child(
                        Button::new("replace", "Replace Messages")
                            .style(ButtonStyle::Filled)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.replace_messages(window, cx);
                            })),
                    ),
            )
        };

        v_flex()
            .id("compact-messages-modal")
            .key_context("CompactMessagesModal")
            .w(rems(40.))
            .elevation_3(cx)
            .on_action(cx.listener(Self::cancel))
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.focus_handle(cx).focus(window, cx);
            }))
            .child(
                Modal::new("compact-messages", None)
                    .header(ModalHeader::new().headline(headline))
                    .when_some(self.last_error.clone(), |this, error| {
                        this.section(
                            Section::new().child(
                                Banner::new()
                                    .severity(Severity::Error)
                                    .child(div().text_xs().child(error)),
                            ),
                        )
                    })
                    .child(content)
                    .footer(footer),
            )
    }
}
