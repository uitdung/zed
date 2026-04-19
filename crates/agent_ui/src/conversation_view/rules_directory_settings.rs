use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_settings::AgentSettings;
use fs::Fs;
use gpui::{DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, SharedString, Window};
use project::{Project, WorktreeId};
use settings::{modify_project_settings_json, RulesDirectoryEntry, Settings, SettingsLocation};
use ui::{
    IconButton, IconName, IconSize, KeyBinding, Modal, ModalFooter, ModalHeader, Section, Switch,
    ToggleState, prelude::*,
};
use util::rel_path::RelPath;
use workspace::{ModalView, Workspace};

pub struct RulesDirectorySettingsModal {
    focus_handle: FocusHandle,
    entries: Vec<RulesDirectoryEntry>,
    worktree_abs_path: PathBuf,
    worktree_id: WorktreeId,
    fs: Arc<dyn Fs>,
}

impl RulesDirectorySettingsModal {
    pub fn toggle(
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut gpui::Context<Workspace>,
        project: Entity<Project>,
    ) {
        let first_worktree = project.read(cx).visible_worktrees(cx).next();
        let Some(worktree) = first_worktree else {
            return;
        };
        let worktree_id = worktree.read(cx).id();
        let worktree_abs_path = worktree.read(cx).abs_path().to_path_buf();
        let fs = <dyn Fs>::global(cx);

        workspace.toggle_modal(window, cx, |window, cx| {
            Self::new(window, cx, worktree_id, worktree_abs_path, fs)
        });
    }

    fn new(
        window: &mut Window,
        cx: &mut gpui::Context<Self>,
        worktree_id: WorktreeId,
        worktree_abs_path: PathBuf,
        fs: Arc<dyn Fs>,
    ) -> Self {
        let location = SettingsLocation {
            worktree_id,
            path: RelPath::empty(),
        };
        let settings = AgentSettings::get(Some(location), cx);
        let entries = settings.rules_directories.clone();
        let focus_handle = cx.focus_handle();

        cx.observe_global_in::<settings::SettingsStore>(window, |this, _window, cx| {
            let location = SettingsLocation {
                worktree_id: this.worktree_id,
                path: RelPath::empty(),
            };
            let settings = AgentSettings::get(Some(location), cx);
            let new_entries = settings.rules_directories.clone();
            if new_entries != this.entries {
                this.entries = new_entries;
                cx.notify();
            }
        })
        .detach();

        Self {
            focus_handle,
            entries,
            worktree_abs_path,
            worktree_id,
            fs,
        }
    }

    fn toggle_entry(&mut self, index: usize, _: &mut Window, cx: &mut gpui::Context<Self>) {
        if let Some(entry) = self.entries.get_mut(index) {
            let current = entry.enabled.unwrap_or(true);
            entry.enabled = Some(!current);
            Self::save_to_settings(
                self.entries.clone(),
                &self.worktree_abs_path,
                &self.fs,
                cx,
            );
            cx.notify();
        }
    }

    fn remove_entry(&mut self, index: usize, _: &mut Window, cx: &mut gpui::Context<Self>) {
        if index < self.entries.len() {
            self.entries.remove(index);
            Self::save_to_settings(
                self.entries.clone(),
                &self.worktree_abs_path,
                &self.fs,
                cx,
            );
            cx.notify();
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, _: &mut Window, cx: &mut gpui::Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn save_to_settings(
        entries: Vec<RulesDirectoryEntry>,
        worktree_abs_path: &Path,
        fs: &Arc<dyn Fs>,
        cx: &mut gpui::Context<Self>,
    ) {
        let fs = fs.clone();
        let worktree_abs_path = worktree_abs_path.to_path_buf();

        cx.spawn(async move |_this, _cx| {
            modify_project_settings_json(&fs, &worktree_abs_path, |json| {
                let Some(root) = json.as_object_mut() else {
                    return false;
                };
                if entries.is_empty() {
                    root.remove("rules_directories");
                } else {
                    let entries_json = serde_json::to_value(&entries).unwrap_or_default();
                    root.insert("rules_directories".to_string(), entries_json);
                }
                true
            })
            .await;
        })
        .detach();
    }
}

impl EventEmitter<DismissEvent> for RulesDirectorySettingsModal {}

impl Focusable for RulesDirectorySettingsModal {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl ModalView for RulesDirectorySettingsModal {}

impl Render for RulesDirectorySettingsModal {
    fn render(&mut self, _window: &mut Window, cx: &mut gpui::Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle(cx);

        v_flex()
            .id("rules-directory-settings-modal")
            .key_context("RulesDirectorySettingsModal")
            .w(rems(36.))
            .elevation_3(cx)
            .on_action(cx.listener(Self::cancel))
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.focus_handle(cx).focus(window, cx);
            }))
            .child(
                Modal::new("rules-directory-settings", None)
                    .header(
                        ModalHeader::new()
                            .headline("Rule Directories")
                            .description(
                                "Directories scanned for rule files (.md, .txt, .mdc) \
                                 included in agent context. Right-click in the Project \
                                 Panel to add more.",
                            ),
                    )
                    .section(if self.entries.is_empty() {
                        Section::new().child(
                            v_flex()
                                .py_6()
                                .items_center()
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(cx.theme().colors().text_muted)
                                        .child("No rule directories configured."),
                                )
                                .child(
                                    div()
                                        .mt_1()
                                        .text_sm()
                                        .text_color(cx.theme().colors().text_muted)
                                        .child(
                                            "Right-click a folder in the Project Panel to add.",
                                        ),
                                ),
                        )
                    } else {
                        Section::new().children(
                            self.entries.iter().enumerate().map(|(index, entry)| {
                                let enabled = entry.enabled.unwrap_or(true);
                                let toggle_state = if enabled {
                                    ToggleState::Selected
                                } else {
                                    ToggleState::Unselected
                                };

                                h_flex()
                                    .py_1()
                                    .gap_2()
                                    .items_center()
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w_0()
                                            .overflow_x_hidden()
                                            .text_sm()
                                            .when(!enabled, |el| {
                                                el.text_color(cx.theme().colors().text_muted)
                                            })
                                            .child(entry.path.clone()),
                                    )
                                    .child(
                                        Switch::new(
                                            SharedString::from(format!("toggle-{index}")),
                                            toggle_state,
                                        )
                                        .on_click(cx.listener(
                                            move |this, _state, window, cx| {
                                                this.toggle_entry(index, window, cx);
                                            },
                                        )),
                                    )
                                    .child(
                                        IconButton::new(
                                            SharedString::from(format!("remove-{index}")),
                                            IconName::Trash,
                                        )
                                        .icon_size(IconSize::Small)
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.remove_entry(index, window, cx);
                                        })),
                                    )
                            }),
                        )
                    })
                    .footer(
                        ModalFooter::new().end_slot(
                            Button::new("close-btn", "Close")
                                .key_binding(
                                    KeyBinding::for_action_in(&menu::Cancel, &focus_handle, cx)
                                        .map(|kb| kb.size(rems_from_px(12.))),
                                )
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.cancel(&menu::Cancel, window, cx);
                                })),
                        ),
                    ),
            )
    }
}
