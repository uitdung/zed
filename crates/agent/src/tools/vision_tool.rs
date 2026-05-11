use std::sync::Arc;

use agent_client_protocol::schema as acp;
use anyhow::{Context as _, Result, bail};
use base64::Engine;
use futures::{AsyncBufReadExt as _, AsyncReadExt as _, FutureExt as _, StreamExt as _, io::BufReader};
use gpui::{App, Task};
use http_client::{AsyncBody, HttpClientWithUrl};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ui::SharedString;

use agent_settings::AgentSettings;
use settings::Settings;

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Analyzes an image using a vision-capable AI model and returns a text description.
///
/// Use this tool when the user attaches or references an image file and the current model
/// cannot process images directly. The tool reads the image file, sends it to a vision
/// model, and returns a detailed textual description that the current model can understand.
///
/// Supports local file paths (absolute) and data URIs (data:image/...;base64,...).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct VisionToolInput {
    /// Absolute path to the image file to analyze (e.g. "/tmp/image.png" or "C:\\Users\\photo.jpg").
    /// Can also be a data URI in the format "data:image/png;base64,<base64_data>".
    image_source: String,
    /// What to analyze, extract, or understand from the image.
    /// Be specific about what information you need.
    prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum VisionToolOutput {
    #[serde(rename = "success")]
    Success { description: String },
    #[serde(rename = "error")]
    Error { error: String },
}

impl From<VisionToolOutput> for language_model::LanguageModelToolResultContent {
    fn from(value: VisionToolOutput) -> Self {
        match value {
            VisionToolOutput::Success { description } => description.into(),
            VisionToolOutput::Error { error } => error.into(),
        }
    }
}

pub struct VisionTool {
    http_client: Arc<HttpClientWithUrl>,
}

impl VisionTool {
    pub fn new(http_client: Arc<HttpClientWithUrl>) -> Self {
        Self { http_client }
    }

    fn read_config(cx: &mut App) -> Result<(String, String, String), VisionToolOutput> {
        let settings = AgentSettings::get_global(cx);

        if let Some(vision_config) = &settings.vision_tool {
            if let Some(api_key) = &vision_config.api_key {
                let api_url = vision_config
                    .api_url
                    .clone()
                    .unwrap_or_else(|| "https://api.z.ai/api/coding/paas/v4/".to_string());
                let model = vision_config
                    .model
                    .clone()
                    .unwrap_or_else(|| "glm-4.6v".to_string());
                return Ok((api_key.clone(), api_url, model));
            }
        }

        let api_key = std::env::var("Z_AI_API_KEY")
            .or_else(|_| std::env::var("ZHIPU_API_KEY"))
            .or_else(|_| std::env::var("ZAI_API_KEY"))
            .map_err(|_| VisionToolOutput::Error {
                error: "Vision tool not configured. Set `vision_tool.api_key` in settings, \
                       or set Z_AI_API_KEY environment variable.".into(),
            })?;

        let api_url = std::env::var("Z_AI_BASE_URL")
            .unwrap_or_else(|_| "https://api.z.ai/api/coding/paas/v4/".to_string());

        let model = std::env::var("Z_AI_VISION_MODEL")
            .unwrap_or_else(|_| "glm-4.6v".to_string());

        Ok((api_key, api_url, model))
    }

    fn encode_image_source(image_source: &str) -> Result<String> {
        if image_source.starts_with("data:image/") {
            log::info!("Vision tool: image_source is already a data URI (len={})", image_source.len());
            return Ok(image_source.to_string());
        }

        if image_source.starts_with("http://") || image_source.starts_with("https://") {
            log::info!("Vision tool: image_source is a URL: {image_source}");
            return Ok(image_source.to_string());
        }

        let path = std::path::Path::new(image_source);
        if !path.exists() {
            log::error!("Vision tool: image file not found: {image_source}");
            bail!("Image file not found: {}", image_source);
        }

        let bytes = std::fs::read(path).context("Failed to read image file")?;
        log::info!(
            "Vision tool: read image file {image_source} ({} bytes)",
            bytes.len()
        );

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png")
            .to_lowercase();

        let mime_type = match extension.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            _ => "image/png",
        };

        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let data_url = format!("data:{mime_type};base64,{encoded}");
        log::info!(
            "Vision tool: encoded to data URI (mime={mime_type}, base64_len={})",
            encoded.len()
        );
        Ok(data_url)
    }

    async fn stream_vision_api(
        http_client: Arc<HttpClientWithUrl>,
        api_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
        image_data_url: &str,
        event_stream: ToolCallEventStream,
    ) -> Result<String> {
        use http_client::http::{Method, Request as HttpRequest};

        let endpoint = format!("{api_url}chat/completions");
        let request_body = serde_json::json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": image_data_url } },
                    { "type": "text", "text": prompt }
                ]
            }],
            "thinking": { "type": "enabled" },
            "stream": true,
            "max_tokens": 32768
        });

        let body_str = serde_json::to_string(&request_body)
            .context("Failed to serialize vision API request")?;

        log::info!(
            "Vision tool: calling POST {endpoint} (model={model}, body_len={})",
            body_str.len()
        );

        let request = HttpRequest::builder()
            .method(Method::POST)
            .uri(&endpoint)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key.trim()))
            .body(AsyncBody::from(body_str))
            .context("Failed to build vision API request")?;

        let response = http_client
            .send(request)
            .await
            .context("Failed to send vision API request")?;

        let status = response.status();
        log::info!("Vision tool: API responded status={status}");

        if !status.is_success() {
            let mut body = Vec::new();
            let mut response = response;
            response
                .body_mut()
                .read_to_end(&mut body)
                .await
                .context("Failed to read vision API error response body")?;

            let text = String::from_utf8_lossy(&body);
            log::error!(
                "Vision API error: endpoint={endpoint}, status={status}, body={text}"
            );
            bail!("Vision API error (status {status}): {text}");
        }

        let reader = BufReader::new(response.into_body());
        let mut lines = reader.lines();

        let mut thinking_text = String::new();
        let mut content_text = String::new();

        while let Some(line_result) = lines.next().await {
            let line = match line_result {
                Ok(line) => line,
                Err(error) => {
                    log::error!("Vision tool: error reading SSE line: {error}");
                    break;
                }
            };

            let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
                continue;
            };

            let data = data.trim();
            if data == "[DONE]" {
                break;
            }

            let event: serde_json::Value = match serde_json::from_str(data) {
                Ok(value) => value,
                Err(error) => {
                    log::warn!("Vision tool: failed to parse SSE JSON: {error}, data: {data}");
                    continue;
                }
            };

            if let Some(error) = event.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()) {
                bail!("Vision API stream error: {error}");
            }

            let choices = match event.get("choices").and_then(|c| c.as_array()) {
                Some(choices) => choices,
                None => continue,
            };

            let Some(choice) = choices.first() else {
                continue;
            };

            let delta = match choice.get("delta") {
                Some(delta) => delta,
                None => continue,
            };

            if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                thinking_text.push_str(reasoning);
            }

            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                content_text.push_str(content);

                let preview = if content_text.len() > 300 {
                    let boundary = content_text.floor_char_boundary(300);
                    format!("{}...", &content_text[..boundary])
                } else {
                    content_text.clone()
                };
                event_stream.update_fields(
                    acp::ToolCallUpdateFields::new()
                        .title("Analyzing image...")
                        .content(vec![acp::ToolCallContent::Content(
                            acp::Content::new(preview),
                        )]),
                );
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                log::info!("Vision tool: stream finished with reason: {finish_reason}");
                break;
            }
        }

        if content_text.is_empty() {
            log::warn!(
                "Vision tool: no content in streaming response. thinking_text len={}, last event state unknown",
                thinking_text.len()
            );
            bail!("Vision model returned no content");
        }

        log::info!(
            "Vision tool: streaming complete. content_len={}, thinking_len={}",
            content_text.len(),
            thinking_text.len()
        );

        Ok(content_text)
    }
}

impl AgentTool for VisionTool {
    type Input = VisionToolInput;
    type Output = VisionToolOutput;

    const NAME: &'static str = "vision";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => {
                let short = if input.image_source.len() > 60 {
                    format!("{}...", &input.image_source[..57])
                } else {
                    input.image_source
                };
                format!("Analyze image: {short}").into()
            }
            Err(_) => "Analyze image".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let http_client = self.http_client.clone();
        let config = Self::read_config(cx);
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| VisionToolOutput::Error {
                error: format!("Failed to receive tool input: {e}"),
            })?;

            let (api_key, api_url, model) = config?;

            let image_data_url =
                Self::encode_image_source(&input.image_source).map_err(|e| {
                    VisionToolOutput::Error {
                        error: format!("Failed to load image: {e}"),
                    }
                })?;

            let result = {
                let http_client = http_client.clone();
                let event_stream = event_stream.clone();
                let prompt = input.prompt.clone();
                let image_data_url = image_data_url.clone();
                let api_url = api_url.clone();
                let api_key = api_key.clone();
                let model = model.clone();

                async move {
                    Self::stream_vision_api(
                        http_client,
                        &api_url,
                        &api_key,
                        &model,
                        &prompt,
                        &image_data_url,
                        event_stream,
                    )
                    .await
                }
            };

            let result = futures::select! {
                result = result.fuse() => result,
                _ = event_stream.cancelled_by_user().fuse() => {
                    return Err(VisionToolOutput::Error {
                        error: "Vision analysis cancelled by user".into(),
                    });
                }
            };

            match result {
                Ok(description) => {
                    event_stream.update_fields(
                        acp::ToolCallUpdateFields::new()
                            .title("Image analyzed")
                            .content(vec![acp::ToolCallContent::Content(
                                acp::Content::new(description.clone()),
                            )]),
                    );
                    Ok(VisionToolOutput::Success { description })
                }
                Err(e) => Err(VisionToolOutput::Error {
                    error: format!("Vision analysis failed: {e}"),
                }),
            }
        })
    }
}
