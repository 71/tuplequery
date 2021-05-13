use std::collections::HashMap;

use crates_index::Index;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use serde_json::Value;
use tuplequery::datascript::{parse_clause, query, Clause, SupportedTuple, Triple, Val};

type SmallVec<'a, const N: usize> = smallvec::SmallVec<[Option<Val<'a>>; 3]>;
type HeapVec<'a> = std::vec::Vec<Option<Val<'a>>>;
type DenseTuple<'a, const N: usize> = tuplequery::tuple::DenseTuple<Val<'a>, N>;
type SparseTuple<'a, const N: usize> = tuplequery::tuple::SparseTuple<Val<'a>, N>;

fn benchmark_datascript(c: &mut Criterion) {
    // crates.io benchmark.
    //
    // This benchmark fetches crates information from crates.io and finds what
    // crates ripgrep depends on.
    {
        // Load crates.io index.
        let index = Index::new_cargo_default();

        assert!(index.exists(), "local crates.io index cannot be found");
        index.retrieve_or_update().unwrap();

        // Convert index to triples.
        let mut triples = Vec::new();
        let crates = index.crates().collect::<Vec<_>>();
        let entity_id_by_crate_name = crates
            .iter()
            .enumerate()
            .map(|(i, x)| (x.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        // As of 2021-05-13, there are 60,950 crates, leading to 1,354,361
        // tuples.
        eprintln!("crates.io: loaded {} crates", entity_id_by_crate_name.len());

        for krate in crates {
            let entity = entity_id_by_crate_name[krate.name()];

            triple(
                &mut triples,
                entity,
                "name",
                Val::Str(krate.name().to_string().into()),
            );

            for &version in &[krate.latest_version()] {
                let version_entity = entity_id_by_crate_name.len() + triples.len();

                triple(&mut triples, entity, "version", Val::Entity(version_entity));

                triple(
                    &mut triples,
                    version_entity,
                    "name",
                    Val::Str(version.version().to_string().into()),
                );

                for dependency in version.dependencies() {
                    let dependency_entity = entity_id_by_crate_name.len() + triples.len();
                    let dependency_crate_id = match entity_id_by_crate_name.get(dependency.name()) {
                        Some(id) => *id,
                        None => continue,
                    };

                    triple(
                        &mut triples,
                        entity,
                        "dependency",
                        Val::Entity(dependency_entity),
                    );
                    triple(
                        &mut triples,
                        version_entity,
                        "dependency",
                        Val::Entity(dependency_entity),
                    );

                    triple(
                        &mut triples,
                        dependency_entity,
                        "crate",
                        Val::Entity(dependency_crate_id),
                    );
                    triple(
                        &mut triples,
                        dependency_entity,
                        "optional?",
                        Val::Bool(dependency.is_optional()),
                    );
                }
            }
        }

        eprintln!("crates.io: loaded {} triples", triples.len());

        // Prepare clauses.
        let clauses = clauses(&[
            r#" ?a        "name"       "ripgrep" "#,
            r#" ?a        "dependency" ?dep      "#,
            r#" ?dep      "crate"      ?depcrate "#,
            r#" ?depcrate "name"       ?depname  "#,
        ]);

        // Run benchmark.
        let mut group = c.benchmark_group("crates.io");

        bench_query_all(&mut group, &clauses, &triples);

        group.finish();
    }
}

fn triple<'a>(triples: &mut Vec<Triple<'a>>, entity: usize, prop: &'static str, value: Val<'a>) {
    triples.push([Val::Entity(entity), Val::Str(prop.into()), value]);
}

fn clauses(clauses: &[&'static str]) -> Vec<Clause> {
    clauses
        .iter()
        .copied()
        .map(parse_clause)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn bench_query_all<'a>(
    group: &mut BenchmarkGroup<WallTime>,
    clauses: &[Clause],
    triples: &[Triple<'a>],
) {
    bench_query::<HeapVec>(group, "vec", clauses, triples);

    bench_query::<SmallVec<3>>(group, "smallvec_3", clauses, triples);
    bench_query::<DenseTuple<3>>(group, "dense_tuple_3", clauses, triples);
    bench_query::<SparseTuple<3>>(group, "sparse_tuple_3", clauses, triples);

    bench_query::<SmallVec<8>>(group, "smallvec_8", clauses, triples);
    bench_query::<DenseTuple<8>>(group, "dense_tuple_8", clauses, triples);
    bench_query::<SparseTuple<8>>(group, "sparse_tuple_8", clauses, triples);
}

fn bench_query<'a, T: SupportedTuple<'a>>(
    group: &mut BenchmarkGroup<WallTime>,
    id: &'static str,
    clauses: &[Clause],
    triples: &[Triple<'a>],
) {
    let show_results = false;
    let clauses = clauses.iter().cloned().collect::<Vec<Clause>>();

    group.bench_function(id, move |b| {
        b.iter(|| {
            let (results, query, variables) = query::<T>(clauses.clone(), triples).unwrap();

            if show_results {
                let results = results
                    .into_iter()
                    .map(|x| x.to_json(query.fields(), &variables))
                    .collect::<Vec<_>>();

                eprintln!("{}", Value::Array(results).to_string());
            }
        })
    });
}

criterion_group!(benches, benchmark_datascript);
criterion_main!(benches);
