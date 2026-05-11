use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match garbelour::run_cli(args) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("garbelour: {:#}", e);
            ExitCode::from(2_u8)
        }
    }
}
