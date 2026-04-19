use std::sync::Arc;

use gpui::{App, Entity, Task};
use project::{Project, ProjectPath};
use prompt_store::{RulesDirContext, RulesDirFileContext};
use settings::{Settings, SettingsLocation};
use collections::{HashMap, HashSet};
use util::rel_path::RelPath;

use agent_settings::AgentSettings;

const RULE_FILE_EXTENSIONS: &[&str] = &["md", "txt", "mdc"];
const MAX_TOTAL_CHARS: usize = 200_000;

pub fn scan_rules_directories(
    project: &Entity<Project>,
    cx: &mut App,
) -> Task<Vec<RulesDirContext>> {
    let mut files_to_read: Vec<(project::WorktreeId, Arc<RelPath>, String, String)> = Vec::new();

    for worktree in project.read(cx).visible_worktrees(cx) {
        let tree = worktree.read(cx);
        let snapshot = tree.snapshot();
        let worktree_id = tree.id();
        let worktree_root_name = tree.root_name_str().to_string();

        let location = SettingsLocation {
            worktree_id,
            path: RelPath::empty(),
        };
        let settings = AgentSettings::get(Some(location), cx);
        let entries = settings.rules_directories.clone();

        if entries.is_empty() {
            continue;
        }

        for entry in &entries {
            if !entry.enabled.unwrap_or(true) {
                continue;
            }

            let Ok(rel_path) = RelPath::unix(&entry.path) else {
                continue;
            };

            let Some(snapshot_entry) = snapshot.entry_for_path(rel_path) else {
                continue;
            };

            if snapshot_entry.is_file() {
                let extension = snapshot_entry.path.extension();
                if extension.map(|ext| RULE_FILE_EXTENSIONS.contains(&ext)).unwrap_or(false) {
                    files_to_read.push((worktree_id, snapshot_entry.path.clone(), entry.path.clone(), worktree_root_name.clone()));
                }
            } else if snapshot_entry.is_dir() {
                for child in snapshot.child_entries(rel_path) {
                    if !child.is_file() {
                        continue;
                    }
                    let Some(extension) = child.path.extension() else {
                        continue;
                    };
                    if !RULE_FILE_EXTENSIONS.contains(&extension) {
                        continue;
                    }

                    files_to_read.push((worktree_id, child.path.clone(), entry.path.clone(), worktree_root_name.clone()));
                }
            }
        }
    }

    let mut seen: HashSet<(project::WorktreeId, String)> = HashSet::default();
    files_to_read.retain(|(worktree_id, path, _, _)| {
        seen.insert((*worktree_id, path.as_unix_str().to_string()))
    });

    if files_to_read.is_empty() {
        return Task::ready(Vec::new());
    }

    let project = project.clone();
    cx.spawn(async move |cx| {
        let mut grouped: HashMap<(String, String), Vec<RulesDirFileContext>> =
            HashMap::default();

        let mut total_chars: usize = 0;

        for (worktree_id, path, dir_path, root_name) in files_to_read {
            let project_path = ProjectPath {
                worktree_id,
                path: path.clone(),
            };

            let buffer_result = project
                .update(cx, |project, cx| project.open_buffer(project_path, cx))
                .await;

            let Ok(buffer) = buffer_result else {
                log::warn!("Failed to open buffer for rules file: {buffer_result:?}");
                continue;
            };

            let text = buffer.read_with(cx, |buffer, _| buffer.as_rope().to_string());
            let text = text.trim().to_string();
            if !text.is_empty() {
                if total_chars + text.len() > MAX_TOTAL_CHARS {
                    log::warn!(
                        "Rules directories char limit ({MAX_TOTAL_CHARS}) reached, skipping remaining files"
                    );
                    break;
                }
                total_chars += text.len();
                grouped
                    .entry((dir_path, root_name))
                    .or_default()
                    .push(RulesDirFileContext {
                        path_in_worktree: path,
                        text,
                    });
            }
        }

        let mut result: Vec<RulesDirContext> = grouped
            .into_iter()
            .map(|((directory_path, worktree_root_name), files)| RulesDirContext {
                worktree_root_name,
                directory_path,
                files,
            })
            .collect();

        result.sort_by(|a, b| (&a.directory_path, &a.worktree_root_name).cmp(&(&b.directory_path, &b.worktree_root_name)));

        result
    })
}
