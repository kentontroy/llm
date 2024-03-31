use chrono::offset::Utc;
use llm_base::InferenceStats;
use serde::{Deserialize, Serialize};
use crate::parameters::ModelParamsFromEnv;

#[derive(Serialize, Deserialize)]
pub struct InferenceLog {
    pub time_as_of_logging: String,
    pub model: String,
    pub temperature: f32,
    pub top_k: usize,
    pub max_tokens_generate: usize,
    pub feed_prompt_duration: f32,
    pub prompt_tokens: usize,
    pub predict_duration: f32,
    pub predict_tokens: usize,
    pub prompt: String,
    pub response: String,
}
pub struct InferenceLogBuilder {
    pub time_as_of_logging: String,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens_generate: Option<usize>, 
    pub feed_prompt_duration: f32,
    pub prompt_tokens: usize,
    pub predict_duration: f32,
    pub predict_tokens: usize,
    pub prompt: String,
    pub response: String,
}

impl InferenceLog {
    pub fn new(stats: &InferenceStats, prompt: &str, response: &str) -> InferenceLogBuilder {
        InferenceLogBuilder {
            time_as_of_logging: format!("{:?}", Utc::now()),
            model: None,
            temperature: None,
            top_k: None,
            max_tokens_generate: None,
            feed_prompt_duration: stats.feed_prompt_duration.as_secs_f32(),
            prompt_tokens: stats.prompt_tokens,
            predict_duration: stats.predict_duration.as_secs_f32(),
            predict_tokens: stats.predict_tokens,
            prompt: prompt.to_string(),
            response: response.to_string(),
        }
    }
}

impl InferenceLogBuilder {
    pub fn model(&mut self, model_path: &str) -> &mut Self {
        self.model = Some(model_path.to_string());
        self
    }
    pub fn parameters(&mut self, params: &ModelParamsFromEnv) -> &mut Self {
        self.temperature = Some(params.temperature.clone());
        self.top_k = Some(params.top_k.clone());
        self.max_tokens_generate = Some(params.max_tokens_generate.clone());
        self
    }
    pub fn build(&self) -> InferenceLog {
        let model = self.model.clone().unwrap_or_default();
        InferenceLog {
            time_as_of_logging: self.time_as_of_logging.clone(),
            model: model.to_owned(),
            temperature: self.temperature.clone().unwrap_or_default(),
            top_k: self.top_k.clone().unwrap_or_default(),
            max_tokens_generate: self.max_tokens_generate.clone().unwrap_or_default(),
            feed_prompt_duration: self.feed_prompt_duration.clone(),
            prompt_tokens: self.prompt_tokens.clone(),
            predict_duration: self.predict_duration.clone(),
            predict_tokens: self.predict_tokens.clone(),
            prompt: self.prompt.clone(),
            response: self.response.clone(),
        }
    }
}