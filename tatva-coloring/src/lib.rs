use ahash::{AHashMap, AHashSet};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Color a sparse matrix's distance-2 graph and emit colors and seed vectors.
#[pyfunction]
fn distance2_color_and_seeds(
    py: Python<'_>,
    row_ptr: PyReadonlyArray1<'_, i64>,
    col_idx: PyReadonlyArray1<'_, i64>,
    n_dofs: usize,
) -> PyResult<(Py<PyArray1<i32>>, Vec<Py<PyArray1<f64>>>)> {
    let row_ptr = row_ptr.as_slice()?;
    let col_idx = col_idx.as_slice()?;

    if row_ptr.len() != n_dofs + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "row_ptr length must be n_dofs + 1",
        ));
    }

    // Build adjacency from CSR.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_dofs];
    for i in 0..n_dofs {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let slice = &col_idx[start..end];
        adjacency[i].extend(slice.iter().map(|&v| v as usize));
    }

    // Build distance-2 adjacency (neighbors and neighbors-of-neighbors).
    let adjacency2 = distance2_adjacency(&adjacency);

    // Greedy coloring on distance-2 adjacency.
    let colors = greedy_color(&adjacency2);

    // Pack outputs for Python: colors as np.int32 and seeds as list of float64.
    let colors_py = PyArray1::from_iter(py, colors.iter().map(|&c| c as i32)).unbind();
    let seeds = seeds_from_colors(py, &colors)?;

    Ok((colors_py, seeds))
}

/// Compute distance-2 adjacency from 1-hop adjacency.
fn distance2_adjacency(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut adj2: Vec<AHashSet<usize>> = vec![AHashSet::new(); n];

    for (i, neighs) in adjacency.iter().enumerate() {
        for &j in neighs {
            if i != j {
                adj2[i].insert(j);
                adj2[j].insert(i);
            }
            for &k in &adjacency[j] {
                if k != i {
                    adj2[i].insert(k);
                    adj2[k].insert(i);
                }
            }
        }
    }

    adj2
        .into_iter()
        .map(|set| {
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Simple greedy coloring: smallest available color per vertex.
fn greedy_color(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut colors = vec![usize::MAX; n];
    let mut used: AHashMap<usize, usize> = AHashMap::new();

    for i in 0..n {
        used.clear();
        for &nb in &adjacency[i] {
            if let Some(&c) = colors.get(nb) {
                if c != usize::MAX {
                    *used.entry(c).or_insert(0) += 1;
                }
            }
        }
        let mut c = 0;
        while used.contains_key(&c) {
            c += 1;
        }
        colors[i] = c;
    }
    colors
}

/// Generate one-hot seeds per color (float64).
fn seeds_from_colors(py: Python<'_>, colors: &[usize]) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    if colors.is_empty() {
        return Ok(Vec::new());
    }
    let max_color = *colors.iter().max().unwrap_or(&0);
    let n_colors = max_color + 1;
    let n = colors.len();

    let mut seeds = Vec::with_capacity(n_colors);
    for color in 0..n_colors {
        let mut seed = vec![0f64; n];
        for (idx, &c) in colors.iter().enumerate() {
            if c == color {
                seed[idx] = 1.0;
            }
        }
        seeds.push(PyArray1::from_vec(py, seed).unbind());
    }
    Ok(seeds)
}

/// Python module definition.
#[pymodule]
fn _tatva_color(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance2_color_and_seeds, m)?)?;
    Ok(())
}
