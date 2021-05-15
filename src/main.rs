//! This program reads a JSON file from stdin and a set of clauses passed as
//! arguments, and outputs the result of running those clauses on the JSON
//! object.
//!
//! See the [`crate::datascript`] module for more information.

use std::{convert::TryInto, error::Error};

use serde_json::Value;
#[cfg(feature = "trace")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tuplequery::{
    datascript::{parse_clause, query, value_to_triples, SupportedTuple, Val},
    tuple::DenseTuple,
};

fn main() -> Result<(), Box<dyn Error>> {
    // Parse clauses from command line.
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut read_csv = false;

    if args[0] == "--csv" {
        read_csv = true;
        args.remove(0);
    }

    let clauses = args
        .iter()
        .map(|x| parse_clause(x))
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

    #[cfg(feature = "trace")]
    let (flame_layer, guard) = tracing_flame::FlameLayer::with_file("./tracing.folded").unwrap();
    #[cfg(feature = "trace")]
    tracing_subscriber::registry().with(flame_layer).init();
    #[cfg(feature = "trace")]
    let span = tracing::span!(tracing::Level::TRACE, "parse input").entered();

    // Parse input triples from stdin.
    #[allow(unused_assignments)] // Needed to extend lifetime of value.
    let (mut records, mut value) = (Vec::new(), None);

    let triples = if read_csv {
        records = csv::Reader::from_reader(std::io::stdin())
            .records()
            .collect::<Result<Vec<_>, _>>()?;

        let mut triples = Vec::new();

        for (i, record) in records.iter().enumerate() {
            let record = record.iter().map(|x| Val::Str(x)).collect::<Vec<_>>();
            let record_len = record.len();

            triples.push(record.try_into().map_err(move |_| {
                format!(
                    "expected record #{} to have 3 values; it had {}",
                    i, record_len
                )
            })?);
        }

        triples
    } else {
        // Convert input object into triples.
        value = Some(serde_json::from_reader(std::io::stdin())?);

        value_to_triples(value.as_ref().unwrap())
    };

    #[cfg(feature = "trace")]
    drop(span);
    #[cfg(feature = "trace")]
    let span = tracing::span!(tracing::Level::TRACE, "query").entered();

    // Run query.
    #[cfg(feature = "trace")]
    let time = std::time::Instant::now();

    let (results, query, variables) = query::<DenseTuple<_, 3>>(clauses, &triples)?;

    #[cfg(feature = "trace")]
    eprintln!("query execution time: {:?}", time.elapsed());

    #[cfg(feature = "trace")]
    drop(span);
    #[cfg(feature = "trace")]
    let span = tracing::span!(tracing::Level::TRACE, "convert results").entered();

    let json_results = results
        .into_iter()
        .map(|x| x.to_json(query.fields(), &variables))
        .collect();

    #[cfg(feature = "trace")]
    drop(span);

    // Display query results.
    serde_json::to_writer(std::io::stdout(), &Value::Array(json_results)).unwrap();

    #[cfg(feature = "trace")]
    drop(guard);

    Ok(())
}
