//! This program reads a JSON file from stdin and a set of clauses passed as
//! arguments, and outputs the result of running those clauses on the JSON
//! object.
//!
//! See the [`crate::datascript`] module for more information.

use std::{borrow::Cow, convert::TryInto, error::Error};

use serde_json::Value;
use tuplequery::{datascript::{SupportedTuple, Val, parse_clause, query, value_to_triples}, tuple::DenseTuple};

fn main() -> Result<(), Box<dyn Error>> {
    // Parse clauses from command line.
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut read_csv = false;

    if args[0] == "--csv" {
        read_csv = true;
        args.remove(0);
    }

    let clauses = args.into_iter()
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

    // Parse input triples from stdin.
    #[allow(unused_assignments)]  // Needed to extend lifetime of value.
    let mut value = None;

    let triples = if read_csv {
        let mut records = Vec::new();
        let mut reader = csv::Reader::from_reader(std::io::stdin());

        for (i, record) in reader.records().enumerate() {
            let record = record?.into_iter().map(|x| Val::Str(Cow::Owned(x.to_string()))).collect::<Vec<_>>();
            let record_len = record.len();

            records.push(record.try_into().map_err(move |_| format!("expected record #{} to have 3 values; it had {}", i, record_len))?);
        }

        records
    } else {
        // Convert input object into triples.
        value = Some(serde_json::from_reader(std::io::stdin())?);

        value_to_triples(value.as_ref().unwrap())
    };

    // Run query.
    let (results, query, variables) = query::<DenseTuple<_, 3>>(clauses, &triples)?;
    let json_results = results.into_iter().map(|x| x.to_json(query.fields(), &variables)).collect();

    // Display query results.
    serde_json::to_writer(std::io::stdout(), &Value::Array(json_results)).unwrap();

    Ok(())
}
