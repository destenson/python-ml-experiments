
use std::collections::HashMap;
use std::fs::{File, read_dir};
use std::path::Path;
use serde_pickle::{from_slice, Value};
use std::io::{Read, BufReader};

#[derive(Debug)]
struct FinancialDataSource {
    raw_data: HashMap<String, Value>,
}

impl PartialEq for FinancialDataSource {
    fn eq(&self, other: &Self) -> bool {
        true
    }
}

impl FinancialDataSource {
    pub fn new() -> Self {
        FinancialDataSource {raw_data: HashMap::new()}
    }

    pub fn insert(&mut self, key: String, value: Value) {
        self.raw_data.insert(key, value);
    }
}



#[derive(Debug)]
enum LoadError {
    IOError(std::io::Error),
    SerdeError(serde_pickle::Error),
}

impl From<std::io::Error> for LoadError {
    fn from(error: std::io::Error) -> Self {
        LoadError::IOError(error)
    }
}

impl From<serde_pickle::Error> for LoadError {
    fn from(error: serde_pickle::Error) -> Self {
        LoadError::SerdeError(error)
    }
}

fn load_data<P: AsRef<Path>>(path: P) -> Result<FinancialDataSource, LoadError> {
    // Load data from pickle files in path
    let mut data = FinancialDataSource::new();
    let files = read_dir(path.as_ref()).unwrap();
    for file in files {
        let file = file.unwrap();
        let path = file.path();

        // Open the pickle file
        let mut file = File::open(file.path())?;
            
        // Read the file into a byte vector
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Log the raw data for debugging
        println!("Raw data from file {:?}: {:?}", path, &buffer[0..100]);
        
        // Deserialize the pickle data
        match from_slice(&buffer, Default::default()) {
            Ok(val) => {
                // Print the deserialized data
                println!("{:?}", &val);
        
                data.insert(path.to_str().unwrap().to_string(), val);
            },
            Err(e) => {
                eprintln!("Failed to deserialize file {:?}: {:?}", path, e);
                return Err(LoadError::SerdeError(e));
            }
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_data() {
        println!("{}", std::env::current_dir().unwrap().display());
        let data = load_data("../../data/").unwrap();
        assert_eq!(data, FinancialDataSource::new());
    }
}
