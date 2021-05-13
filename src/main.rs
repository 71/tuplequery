//! This program reads a JSON file from stdin and a set of clauses passed as
//! arguments, and outputs the result of running those clauses on the JSON
//! object.
//!
//! See the [`crate::datascript`] module for more information.

use std::error::Error;

use serde_json::Value;
use tuplequery::{
    datascript::{parse_clause, query, value_to_triples, SupportedTuple},
    tuple::DenseTuple,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Parse clauses from command line.
    let clauses = std::env::args()
        .skip(1)
        .map(|x| parse_clause(&x))
        .collect::<Result<Vec<_>, _>>()?;

    if clauses.is_empty() {
        let name = std::env::args().next().unwrap();

        eprintln!(
            r#"
USAGE:
    command | {} [CLAUSES ...]

EXAMPLE:
    curl file.json | {} '?a "age" ?age' '?b "age" ?age'"#,
            name, name
        );

        std::process::exit(1);
    }

    // Parse input object from stdin.
    let value = serde_json::from_reader(std::io::stdin())?;

    // Convert input object into triples.
    let triples = value_to_triples(&value);

    // Run query.
    let (results, query, variables) = query::<DenseTuple<_, 3>>(clauses, &triples)?;
    let json_results = results.into_iter().map(|x| x.to_json(query.fields(), &variables)).collect();

    // Display query results.
    serde_json::to_writer(std::io::stdout(), &Value::Array(json_results)).unwrap();

    Ok(())
}
