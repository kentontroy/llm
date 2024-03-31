use clap::Parser;
use std::{
    convert::Infallible, 
    io::Write, 
    path::PathBuf
};

mod database;
mod parameters;
use crate::database::InferenceLog;

#[derive(Parser)]
struct Args {
    model_path: PathBuf,
    #[arg(long, short = 'p')]
    prompt: Option<String>,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
}
impl Args {
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
}

fn main() {
    let args = Args::parse();
    let tokenizer_source = args.to_tokenizer_source();
    let model_path = args.model_path;
    let prompt = args
        .prompt
        .as_deref()
        .unwrap_or("Rust is a cool programming language because");

    let now = std::time::Instant::now();

// Custom changes made to github branch
// Specify GPU usage
    let mut model_params = llm::ModelParameters::default();
    model_params.use_gpu = true;
    model_params.gpu_layers = Some(10);

    let model = llm::load(
        &model_path,
        tokenizer_source,
        model_params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load model from {model_path:?}: {err}"));

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    let mut session = model.start_session(Default::default());

// Modified the SampleChainBuilder to enable reading parameter values from environment variables
    let params = parameters::ModelParamsFromEnv::default();
    let parameters_from_env = parameters::create_inference_parameters_from_env(&params);

    let mut response_buffer: String = "".to_owned();

    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &parameters_from_env,
            play_back_previous_tokens: false,
            maximum_token_count: Some(params.max_tokens_generate.clone()),
        },
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();
                response_buffer.push_str(t.as_str());
                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );

    match res {
        Ok(result) => {
            let inference_log = InferenceLog::new(&result, prompt, response_buffer.as_str())
                .model(String::from(model_path.to_string_lossy()).as_str())
                .parameters(&params)
                .build(); 

            let log_json = serde_json::to_string_pretty(&inference_log);
            match log_json {
                Ok(json) =>  println!("\n\n{json}"),
                Err(err) => println!("{err}"),
            }
        },
        Err(err) => println!("\n{err}"),
    }
}
