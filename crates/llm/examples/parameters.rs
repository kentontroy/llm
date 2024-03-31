use dotenv::dotenv;
use std::{
    env,
    str::FromStr,
    sync::{Arc, Mutex},
};

#[derive(Copy, Clone)]
pub struct ModelParamsFromEnv {
    pub temperature: f32,
    pub top_k: usize,
    pub max_tokens_generate: usize,
}

impl Default for ModelParamsFromEnv {
    fn default() -> Self {
        Self { 
            temperature: default_temperature(),
            top_k: default_top_k(),
            max_tokens_generate: default_max_tokens_generate(),
        }
    }
}

fn default_temperature() -> f32 {
    dotenv().ok();
    let default_val: f32 = 0.8;
    let temp = env::var("MODEL_PARAM_TEMPERATURE");
    match temp {
        Ok(temp) => {
            println!("Read temperature from env: {}", temp);
            if let Err(e) = f32::from_str(&temp) {
                println!("Failed conversion to f32: {e}. Using default temperature: {default_val}");
                return default_val;
            }
            let res = f32::from_str(&temp).unwrap();
            if res <= 0.0 || res > 1.0 {
                println!("Bad parameter setting. Temperature should be > 0 and <= 1.0. Using default temperature: {default_val}");
                return default_val;
            }
            return res;
        },
        Err(_) => {
            println!("Couldn't read temperature from env. Using default value {default_val}");
            return default_val;
        }
    }
}

fn default_top_k() -> usize {
    dotenv().ok();
    let default_val: usize = 40;
    let top_k = env::var("MODEL_PARAM_TOP_K");
    match top_k {
        Ok(top_k) => {
            println!("Read top_k from env: {}", top_k);
            if let Err(e) = usize::from_str(&top_k) {
                println!("Failed conversion to usize: {e}. Using default top_k: {default_val}");
                return default_val;
            }
            let res = usize::from_str(&top_k).unwrap();
            if res < 20 || res > 80 {
                println!("Bad parameter setting. top_k should be >= 20 and <= 80. Using default top_k: {default_val}");
                return default_val;
            }
            return res;
        },
        Err(_) => {
            println!("Couldn't read top_k from env. Using default value {default_val}");
            return default_val;
        }
    }
}

fn default_max_tokens_generate() -> usize {
    dotenv().ok();
    let default_val: usize = 100;
    let max_tokens_generate = env::var("MAXIMUM_TOKENS_GENERATE");
    match max_tokens_generate {
        Ok(max_tokens_generate) => {
            println!("Read max_tokens_generate from env: {}", max_tokens_generate);
            if let Err(e) = usize::from_str(&max_tokens_generate) {
                println!("Failed conversion to usize: {e}. Using default max_tokens_generate: {default_val}");
                return default_val;
            }
            let res = usize::from_str(&max_tokens_generate).unwrap();
            if res < 30 || res > 10000 {
                println!("Bad parameter setting. max_tokens_generate should be >= 100 and <= 10000. Using default max_tokens_generate: {default_val}");
                return default_val;
            }
            return res;
        },
        Err(_) => {
            println!("Couldn't read max_tokens_generate from env. Using default value {default_val}");
            return default_val;
        }
    }
}

use llm_base::samplers::ConfiguredSamplers;
use llm_base::samplers::llm_samplers::configure::{ SamplerSlot, SamplerChainBuilder };
use llm_base::samplers::llm_samplers::samplers::*;
use llm::InferenceParameters;

pub fn create_inference_parameters_from_env(params: &ModelParamsFromEnv) -> InferenceParameters {
    let mut result = parameterize(params.clone());
    result.ensure_default_slots();
    InferenceParameters {
        sampler: Arc::new(Mutex::new(result.builder.into_chain()))
    }
}

fn parameterize(params: ModelParamsFromEnv) -> ConfiguredSamplers {
    ConfiguredSamplers {
        builder: SamplerChainBuilder::from([
            (
                "repetition",
                SamplerSlot::new_chain(
                    || Box::new(SampleRepetition::default().penalty(1.30).last_n(64)),
                    [],
                ),
            ),
            (
                "freqpresence",
                SamplerSlot::new_chain(
                    || Box::new(SampleFreqPresence::default().last_n(64)),
                    [],
                ),
            ),
            (
                "seqrepetition",
                SamplerSlot::new_chain(|| Box::<SampleSeqRepetition>::default(), []),
            ),
            (
                "topk",
                SamplerSlot::new_single(
                    move || Box::new(SampleTopK::default().k(params.top_k)),
                    Option::<SampleTopK>::None,
                ),
            ),
            (
                "tailfree",
                SamplerSlot::new_single(
                    || Box::<SampleTailFree>::default(),
                    Option::<SampleTailFree>::None,
                ),
            ),
            (
                "locallytypical",
                SamplerSlot::new_single(
                    || Box::<SampleLocallyTypical>::default(),
                    Option::<SampleLocallyTypical>::None,
                ),
            ),
            (
                "topp",
                SamplerSlot::new_single(
                    || Box::new(SampleTopP::default().p(0.95)),
                    Option::<SampleTopP>::None,
                ),
            ),
            (
                "topa",
                SamplerSlot::new_single(
                    || Box::new(SampleTopA::default().a1(0.0).a2(0.0)),
                    Option::<SampleTopA>::None,
                ),
            ),
            (
                "minp",
                SamplerSlot::new_single(
                    || Box::new(SampleMinP::default().p(0.0)),
                    Option::<SampleMinP>::None,
                ),
            ),
            (
                "temperature",
                SamplerSlot::new_single(
                    move || Box::new(SampleTemperature::default().temperature(params.temperature)),
                    Option::<SampleTemperature>::None,
                ),
            ),
            (
                "mirostat1",
                SamplerSlot::new_single(
                    || Box::<SampleMirostat1>::default(),
                    Option::<SampleMirostat1>::None,
                ),
            ),
            (
                "mirostat2",
                SamplerSlot::new_single(
                    || Box::<SampleMirostat2>::default(),
                    Option::<SampleMirostat2>::None,
                ),
            ),
        ]),
        mirostat1: false,
        mirostat2: false,
        incompat_mirostat: false,
    }
}