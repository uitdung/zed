use std::sync::Arc;

use anyhow::Result;
use acp_thread::UserMessageId;
use collections::{HashMap, IndexMap};
use futures::StreamExt;
use gpui::AsyncApp;
use language_model::{
    CompletionIntent, LanguageModel, LanguageModelCompletionEvent, LanguageModelRequest,
    LanguageModelRequestMessage, LanguageModelToolResultContent, Role,
};
use settings::Settings;
use util::markdown::MarkdownCodeBlock;

use crate::thread::{AgentMessage, AgentMessageContent, Message, UserMessage, UserMessageContent};

/// Number of recent agent messages whose tool results should never be compacted.
const UNCOMPACTED_RECENT_AGENT_MESSAGES: usize = 3;

/// Find the message index from which content falls within `chars_threshold` characters from the end.
/// Returns `messages.len()` when threshold is 0 (nothing within range).
/// Returns 0 when total content is smaller than threshold (everything is within range).
pub(crate) fn compute_recent_start_index(
    messages: &[LanguageModelRequestMessage],
    chars_threshold: usize,
) -> usize {
    if chars_threshold == 0 {
        return messages.len();
    }
    let mut chars_from_end = 0usize;
    for i in (0..messages.len()).rev() {
        chars_from_end += messages[i].string_contents().len();
        if chars_from_end >= chars_threshold {
            return i;
        }
    }
    0
}

/// Compact old messages by stripping tool content beyond a threshold from the end.
///
/// Two zones based on distance from the end:
/// - Within `deep_omit_threshold_chars` from end: fully protected (no changes)
/// - Before `deep_omit_threshold_chars` from end: ToolUse and ToolResult content
///   removed entirely, Thinking content omitted, empty messages cleaned up and
///   consecutive same-role merged
///
/// Additionally, the last 3 agent messages with tool results are always protected.
/// Error results are never removed.
pub(crate) fn compact_old_tool_results(
    messages: &mut Vec<LanguageModelRequestMessage>,
    deep_omit_threshold_chars: usize,
) {
    let tool_result_msg_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, msg)| {
            msg.content
                .iter()
                .any(|c| matches!(c, language_model::MessageContent::ToolResult(_)))
        })
        .map(|(i, _)| i)
        .collect();

    if tool_result_msg_indices.is_empty() {
        return;
    }

    let compactible_end = tool_result_msg_indices
        .len()
        .saturating_sub(UNCOMPACTED_RECENT_AGENT_MESSAGES);

    let deep_omit_start_index = if deep_omit_threshold_chars > 0 {
        compute_recent_start_index(messages, deep_omit_threshold_chars)
    } else {
        0
    };

    if deep_omit_start_index == 0 {
        return;
    }

    let protected_msg_indices: Vec<usize> = tool_result_msg_indices
        .iter()
        .skip(compactible_end)
        .copied()
        .collect();

    for i in 0..deep_omit_start_index {
        if protected_msg_indices.contains(&i) {
            continue;
        }

        let msg = &mut messages[i];
        msg.content.retain(|c| match c {
            language_model::MessageContent::ToolUse(_) => false,
            language_model::MessageContent::ToolResult(tool_result) => tool_result.is_error,
            _ => true,
        });
        for content in &mut msg.content {
            if let language_model::MessageContent::Thinking { text, .. } = content {
                let annotation = format!(
                    "[thinking omitted - was {} characters]\n",
                    text.chars().count()
                );
                if text.len() > annotation.len() {
                    *text = annotation;
                }
            }
        }
    }

    messages.retain(|msg| !msg.contents_empty());

    let mut i = 1;
    while i < messages.len() {
        let (left, right) = messages.split_at_mut(i);
        let prev = &mut left[i - 1];
        let curr = &mut right[0];
        if curr.role == prev.role {
            prev.content.append(&mut curr.content);
            if !prev.cache {
                prev.cache = curr.cache;
            }
            if prev.reasoning_details.is_none() {
                prev.reasoning_details = curr.reasoning_details.take();
            }
            messages.remove(i);
        } else {
            i += 1;
        }
    }
}

pub(crate) fn format_request_message_as_markdown(msg: &LanguageModelRequestMessage) -> String {
    let role_header = match msg.role {
        Role::System => "## System",
        Role::User => "## User",
        Role::Assistant => "## Assistant",
    };

    let mut markdown = String::new();
    markdown.push_str(role_header);
    if msg.cache {
        markdown.push_str(" [cached]");
    }
    markdown.push_str("\n\n");

    for content in &msg.content {
        match content {
            language_model::MessageContent::Text(text) => {
                markdown.push_str(text);
                markdown.push('\n');
            }
            language_model::MessageContent::Thinking { text, .. } => {
                markdown.push_str("<thinking>\n");
                markdown.push_str(text);
                markdown.push_str("\n</thinking>\n");
            }
            language_model::MessageContent::RedactedThinking(_) => {
                markdown.push_str("<redacted_thinking />\n");
            }
            language_model::MessageContent::Image(_) => {
                markdown.push_str("<image />\n");
            }
            language_model::MessageContent::ToolUse(tool_use) => {
                markdown.push_str(&format!(
                    "**Tool Use**: {} (ID: {})\n",
                    tool_use.name, tool_use.id
                ));
                markdown.push_str(&format!(
                    "{}\n",
                    MarkdownCodeBlock {
                        tag: "json",
                        text: &format!("{:#}", tool_use.input)
                    }
                ));
            }
            language_model::MessageContent::ToolResult(tool_result) => {
                markdown.push_str(&format!(
                    "**Tool Result**: {} (ID: {})\n",
                    tool_result.tool_name, tool_result.tool_use_id
                ));
                if tool_result.is_error {
                    markdown.push_str("**ERROR:**\n");
                }
                match &tool_result.content {
                    LanguageModelToolResultContent::Text(text) => {
                        markdown.push_str(text);
                        markdown.push_str("\n\n");
                    }
                    LanguageModelToolResultContent::Image(_) => {
                        markdown.push_str("<image />\n\n");
                    }
                }
            }
        }
    }

    markdown.push('\n');
    markdown
}

/// Approximate characters per token, used for estimating token counts from char counts.
pub(crate) const CHARS_PER_TOKEN: usize = 4;

/// Prompt prefix placed before the conversation content when requesting a summary.
pub(crate) const COMPACT_PROMPT_PREFIX: &str = indoc::indoc! {"
    ## Conversation to summarize

    The following is an earlier portion of a conversation between a user and an AI coding assistant.
    Tool outputs have been truncated or omitted, and thinking has been removed.

"};

/// Prompt suffix placed after the conversation content, reinforcing the summarization instruction
/// so the model does not mistake the last message as a prompt to respond to.
pub(crate) const COMPACT_PROMPT_SUFFIX: &str = indoc::indoc! {"

    ---

    INSTRUCTION: Summarize the conversation above (excluding any Recent Context section).
    Do NOT respond to it as an assistant.

    Focus on the **narrative flow** of the conversation — what was discussed, decided, and why.
    The summary will be used as context for continuing this conversation, so prioritize information
    that remains relevant regardless of how files or errors change later.

    Cover these aspects in chronological order:

    1. **User's intent and requests** — what the user wanted to accomplish, stated goals,
       constraints, and preferences expressed during the conversation
    2. **Discussion and decisions** — key design choices made, alternatives considered and why
       they were accepted or rejected, any agreements or disagreements between user and assistant
    3. **Conversation trajectory** — how the topic evolved, what led to what, whether the user
       redirected or refined their requests, and how the scope changed over time
    4. **Important context to preserve** — domain-specific terminology, project conventions,
       user-styled naming or coding preferences, references to prior decisions from before this
       conversation that were mentioned

    Avoid summarizing transient technical state that will be stale by the time this summary is read:
    file contents, error messages, line numbers, variable values, test results. These can be
    re-examined from the codebase directly when needed.

    Target approximately 5000 tokens. Write in clear prose — not just bullet lists.
"};

/// Configurable thresholds for compaction behavior.
/// Stored on Thread, editable from the UI.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Approximate tokens from end beyond which deep omit occurs (Tier 1: tool result stripping).
    pub deep_omit_threshold_tokens: u64,
    /// Approximate tokens from end beyond which auto compact triggers (Tier 2: message summarization).
    /// Content before this boundary is eligible for summarization.
    pub summary_threshold_tokens: u64,
    /// Minimum formatted content length (chars) to trigger auto compact.
    /// Content below this threshold is considered too small to warrant a summary.
    pub min_content_chars: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            deep_omit_threshold_tokens: 80_000,
            summary_threshold_tokens: 80_000,
            min_content_chars: 60_000,
        }
    }
}

impl CompactionConfig {
    pub fn from_settings(cx: &gpui::App) -> Self {
        use agent_settings::AgentSettings;
        let default = Self::default();
        let compaction = AgentSettings::get_global(cx).compaction.as_ref();
        Self {
            deep_omit_threshold_tokens: compaction
                .and_then(|c| c.deep_omit_threshold_tokens)
                .unwrap_or(default.deep_omit_threshold_tokens),
            summary_threshold_tokens: compaction
                .and_then(|c| c.summary_threshold_tokens)
                .unwrap_or(default.summary_threshold_tokens),
            min_content_chars: compaction
                .and_then(|c| c.min_content_chars)
                .unwrap_or(default.min_content_chars),
        }
    }
}

/// Plan produced by [`prepare_message_compaction`] describing what should be summarized.
pub(crate) struct CompactionPlan {
    pub model: Arc<dyn LanguageModel>,
    pub request: LanguageModelRequest,
    pub boundary_index: usize,
}

fn build_compaction_request(
    messages: &[Message],
    model: &Arc<dyn LanguageModel>,
) -> Result<(usize, LanguageModelRequest)> {
    let max_formatted_chars = (model.max_token_count() as usize * CHARS_PER_TOKEN)
        .saturating_sub(
            COMPACT_PROMPT_PREFIX.len()
                + COMPACT_PROMPT_SUFFIX.len()
                + 12_000,
        );

    let mut formatted = String::new();
    for msg in messages {
        let entry = msg.to_markdown();
        if !formatted.is_empty() && formatted.len() + entry.len() > max_formatted_chars {
            break;
        }
        formatted.push_str(&entry);
    }

    if formatted.is_empty() {
        anyhow::bail!("No content to compact");
    }

    let formatted_len = formatted.len();

    let mut full_content = format!("{COMPACT_PROMPT_PREFIX}{formatted}");
    full_content.push_str(COMPACT_PROMPT_SUFFIX);

    let request = LanguageModelRequest {
        intent: Some(CompletionIntent::ThreadContextSummarization),
        messages: vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![full_content.into()],
            cache: false,
            reasoning_details: None,
        }],
        ..Default::default()
    };

    Ok((formatted_len, request))
}

pub(crate) async fn stream_compaction_summary(
    model: Arc<dyn LanguageModel>,
    request: LanguageModelRequest,
    mut cancellation_rx: Option<&mut watch::Receiver<bool>>,
    cx: &mut AsyncApp,
) -> Result<String> {
    let mut summary_text = String::new();
    let mut stream = model.stream_completion(request, cx).await?;

    while let Some(event) = stream.next().await {
        if let Some(cancellation_rx) = cancellation_rx.as_mut() {
            if *cancellation_rx.borrow() {
                log::info!("compact: cancelled during summarization");
                return Ok(summary_text);
            }
        }
        match event {
            Ok(LanguageModelCompletionEvent::Text(text)) => summary_text.push_str(&text),
            Ok(LanguageModelCompletionEvent::Stop(_)) => break,
            Ok(LanguageModelCompletionEvent::UsageUpdate(usage)) => {
                log::info!("compact: input={}, output={}", usage.input_tokens, usage.output_tokens);
            }
            Ok(_) => {}
            Err(err) => return Err(err.into()),
        }
    }

    Ok(summary_text)
}

/// Calculate the string_contents length of a request message after Tier 1 stripping.
/// ToolUse and non-error ToolResult are removed; Thinking is replaced with an annotation
/// (only when the annotation is shorter than the original).
fn stripped_content_len(msg: &LanguageModelRequestMessage) -> usize {
    let mut len = 0;
    for content in &msg.content {
        match content {
            language_model::MessageContent::Text(text) => len += text.len(),
            language_model::MessageContent::Thinking { text, .. } => {
                let annotation_len = format!("[thinking omitted - was {} characters]\n", text.chars().count()).len();
                len += annotation_len.min(text.len());
            }
            language_model::MessageContent::RedactedThinking(_) => {}
            language_model::MessageContent::ToolResult(tool_result) => {
                if tool_result.is_error {
                    if let Some(text) = tool_result.content.to_str() {
                        len += text.len();
                    }
                }
            }
            language_model::MessageContent::ToolUse(_) | language_model::MessageContent::Image(_) => {}
        }
    }
    len
}

/// Shared intermediate result of analyzing conversation messages for compaction.
/// Used by both [`prepare_message_compaction`] and [`compute_compaction_debug_info`]
/// to avoid duplicating the analysis logic.
struct CompactionZones {
    /// Number of request messages produced from the stored messages.
    request_message_count: usize,
    /// Maps each request message index back to the stored message index.
    /// Only populated when `build_stored_mapping` is true.
    request_to_stored: Vec<usize>,
    /// LLM-visible length for each request message (after Tier 1 stripping).
    visible_lens: Vec<usize>,
    /// Index in request_messages where the summary boundary falls.
    boundary_req_idx: usize,
    /// Total chars of all request messages before Tier 1 stripping.
    total_original_chars: usize,
    /// Total chars after Tier 1 stripping (what the LLM actually sees).
    total_visible_chars: usize,
}

/// Analyzes the conversation messages and computes the shared compaction zone data:
/// request message counts, Tier 1 stripping, visible lengths, and summary boundary.
fn analyze_compaction_zones(
    messages: &[Message],
    config: &CompactionConfig,
    build_stored_mapping: bool,
) -> CompactionZones {
    let mut request_messages: Vec<LanguageModelRequestMessage> = Vec::new();
    let mut request_to_stored: Vec<usize> = Vec::new();
    for (stored_idx, msg) in messages.iter().enumerate() {
        for req_msg in msg.to_request() {
            request_messages.push(req_msg);
            if build_stored_mapping {
                request_to_stored.push(stored_idx);
            }
        }
    }

    let deep_omit_chars = (config.deep_omit_threshold_tokens as usize) * CHARS_PER_TOKEN;
    let deep_omit_start_index = if deep_omit_chars > 0 {
        compute_recent_start_index(&request_messages, deep_omit_chars)
    } else {
        0
    };

    let total_original_chars: usize = request_messages
        .iter()
        .map(|m| m.string_contents().len())
        .sum();

    let visible_lens: Vec<usize> = request_messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            if deep_omit_start_index > 0 && i < deep_omit_start_index {
                stripped_content_len(msg)
            } else {
                msg.string_contents().len()
            }
        })
        .collect();

    let total_visible_chars: usize = visible_lens.iter().sum();

    let summary_chars = (config.summary_threshold_tokens as usize) * CHARS_PER_TOKEN;
    let mut chars_from_end = 0usize;
    let mut boundary_req_idx = request_messages.len();
    for i in (0..request_messages.len()).rev() {
        chars_from_end += visible_lens[i];
        if chars_from_end >= summary_chars {
            boundary_req_idx = i;
            break;
        }
    }

    CompactionZones {
        request_message_count: request_messages.len(),
        request_to_stored,
        visible_lens,
        boundary_req_idx,
        total_original_chars,
        total_visible_chars,
    }
}

/// Analyzes the conversation messages and, if enough old content exists, builds a
/// summarization request for the model. Returns `Ok(None)` when compaction should be
/// skipped (no model, already summarized, not enough content, boundary too small, etc.).
pub(crate) fn prepare_message_compaction(
    messages: &[Message],
    summarization_model: Option<Arc<dyn LanguageModel>>,
    config: &CompactionConfig,
    turn_start_index: Option<usize>,
) -> Result<Option<CompactionPlan>> {
    let Some(model) = summarization_model else {
        log::info!("compact: no summarization model configured, skipping");
        return Ok(None);
    };

    let zones = analyze_compaction_zones(messages, config, true);

    if zones.boundary_req_idx == 0 || zones.boundary_req_idx >= zones.request_message_count {
        log::info!("compact: no compactable zone found (messages={}, req={}), skipping", messages.len(), zones.request_message_count);
        return Ok(None);
    }

    let content_to_compact_chars: usize = zones.visible_lens[..zones.boundary_req_idx].iter().sum();
    if content_to_compact_chars < config.min_content_chars {
        log::info!("compact: content too small ({} chars < {}), skipping", content_to_compact_chars, config.min_content_chars);
        return Ok(None);
    }

    let mut boundary_index = zones.request_to_stored[zones.boundary_req_idx];
    if let Some(max_index) = turn_start_index {
        boundary_index = boundary_index.min(max_index);
    }

    if boundary_index == 0 {
        log::info!("compact: boundary capped to 0 by turn_start_index, skipping");
        return Ok(None);
    }

    log::info!(
        "compact: messages={}, boundary={}, content_chars={} chars",
        messages.len(), boundary_index, content_to_compact_chars
    );

    let messages_to_compact = &messages[..boundary_index];
    let (_, request) = match build_compaction_request(messages_to_compact, &model) {
        Ok(result) => result,
        Err(err) => {
            log::info!("compact: failed to build compaction request ({err}), skipping");
            return Ok(None);
        }
    };

    Ok(Some(CompactionPlan {
        model,
        request,
        boundary_index,
    }))
}

/// Replaces messages `[0..boundary_index)` with a summary user message and an
/// agent acknowledgment, cleaning up per-request token usage entries for removed
/// user messages.
pub(crate) fn apply_message_compaction(
    messages: &mut Vec<Message>,
    request_token_usage: &mut HashMap<UserMessageId, language_model::TokenUsage>,
    boundary_index: usize,
    summary_text: String,
) {
    for message in messages.drain(..boundary_index) {
        if let Message::User(user_msg) = message {
            request_token_usage.remove(&user_msg.id);
        }
    }

    messages.insert(0, Message::User(UserMessage {
        id: UserMessageId::new(),
        content: vec![UserMessageContent::Text(format!(
            "[Earlier conversation summary]\n{}",
            summary_text
        ))],
    }));
    messages.insert(1, Message::Agent(AgentMessage {
        content: vec![AgentMessageContent::Text(
            "Understood. I'll keep this summary in mind for context in our continued conversation.".to_string()
        )],
        tool_results: IndexMap::default(),
        reasoning_details: None,
    }));
}

/// Prepare a compaction request for a specific range of messages.
/// Used by the manual compaction UI.
pub(crate) fn prepare_manual_compaction(
    messages: &[Message],
    range: std::ops::Range<usize>,
    model: Arc<dyn LanguageModel>,
) -> Result<CompactionPlan> {
    if range.start >= range.end || range.end > messages.len() {
        anyhow::bail!("Invalid message range: {}..{}", range.start, range.end);
    }

    let messages_to_compact = &messages[range.clone()];
    let (_formatted_len, request) = build_compaction_request(messages_to_compact, &model)?;

    Ok(CompactionPlan {
        model,
        request,
        boundary_index: range.end,
    })
}

/// Debug info for compaction state, used by the UI to display real-time metrics.
#[derive(Debug, Clone)]
pub struct CompactionDebugInfo {
    /// Total chars of all request messages before Tier 1 stripping.
    pub total_original_chars: usize,
    /// Total chars after Tier 1 stripping (what the LLM actually sees).
    pub total_visible_chars: usize,
    /// Chars removed by Tier 1 stripping (original - visible).
    pub stripped_chars: usize,
    /// Chars of content that would be eligible for summarization (before summary boundary).
    pub compactable_chars: usize,
    /// Chars of recent content that is protected (after summary boundary).
    pub recent_chars: usize,
    /// The min_content_chars threshold.
    pub min_content_chars: usize,
    /// The summary_threshold converted to chars.
    pub summary_threshold_chars: usize,
    /// Whether auto_compact is enabled.
    pub auto_compact_enabled: bool,
    /// Number of stored messages.
    pub message_count: usize,
    /// Number of request messages (1 stored message can produce multiple request messages).
    pub request_message_count: usize,
    /// Whether there's enough content to trigger compaction.
    pub would_compact: bool,
}

/// Calculate compaction debug info for the current conversation state.
pub fn compute_compaction_debug_info(
    messages: &[Message],
    config: &CompactionConfig,
    auto_compact_enabled: bool,
) -> CompactionDebugInfo {
    let summary_threshold_chars = (config.summary_threshold_tokens as usize) * CHARS_PER_TOKEN;

    let zones = analyze_compaction_zones(messages, config, false);

    let stripped_chars = zones.total_original_chars.saturating_sub(zones.total_visible_chars);

    let (compactable_chars, recent_chars) =
        if zones.boundary_req_idx == 0 || zones.boundary_req_idx >= zones.request_message_count {
            (0, zones.total_visible_chars)
        } else {
            let compactable: usize = zones.visible_lens[..zones.boundary_req_idx].iter().sum();
            let recent: usize = zones.visible_lens[zones.boundary_req_idx..].iter().sum();
            (compactable, recent)
        };

    let would_compact = auto_compact_enabled
        && zones.boundary_req_idx > 0
        && zones.boundary_req_idx < zones.request_message_count
        && compactable_chars >= config.min_content_chars;

    CompactionDebugInfo {
        total_original_chars: zones.total_original_chars,
        total_visible_chars: zones.total_visible_chars,
        stripped_chars,
        compactable_chars,
        recent_chars,
        min_content_chars: config.min_content_chars,
        summary_threshold_chars,
        auto_compact_enabled,
        message_count: messages.len(),
        request_message_count: zones.request_message_count,
        would_compact,
    }
}