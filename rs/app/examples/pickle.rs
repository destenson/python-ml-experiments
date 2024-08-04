use serde_pickle::{from_reader, DeOptions, Error, Value};
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Error> {
    println!("Current directory: {:?}", std::env::current_dir()?);
    let file = File::open("data/AACG_2023-01-01_2024-08-03.pickle")?;
    let reader = BufReader::new(file);

    let mut options = DeOptions::default();
    // Attempt to deserialize the pickle file
    match from_reader(reader, options) {
        Ok(data) => {
            let data: Value = data;
            println!("Deserialized data: {:?}", data);
        }
        Err(e) => {
            match e {
                Error::Syntax(error_code) => {
                    match error_code {
                        serde_pickle::error::ErrorCode::UnresolvedGlobal => {
                            eprintln!("Unresolved global reference in pickle file.");
                        }
                        _ => {
                            eprintln!("Other syntax error: {:?}", error_code);
                        }
                    }
                }
                _ => {
                    eprintln!("Other deserialization error: {:?}", e);
                }
            }
        }
    }

    Ok(())
}