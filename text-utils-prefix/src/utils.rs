use itertools::Itertools;

pub fn optimized_continuation_permutation<C>(continuations: &[C]) -> (Vec<usize>, Vec<usize>)
where
    C: AsRef<[u8]>,
{
    let permutation: Vec<_> = continuations
        .iter()
        .enumerate()
        .sorted_by(|(_, a), (_, b)| a.as_ref().cmp(b.as_ref()))
        .map(|(i, _)| i)
        .collect();
    let mut skips = vec![0; continuations.len()];
    for i in 0..permutation.len() {
        // if the current key is a prefix of the next one, we can skip the
        // latter
        let continuation = continuations[permutation[i]].as_ref();
        while let Some(next) = permutation.get(i + skips[i] + 1) {
            let next_continuation = continuations[*next].as_ref();
            if next_continuation.starts_with(continuation) {
                skips[i] += 1;
            } else {
                break;
            }
        }
    }
    (permutation, skips)
}

#[cfg(test)]
mod test {
    use crate::utils::optimized_continuation_permutation;

    #[test]
    fn test_optimized_continuation_permutation() {
        let items = ["de", "a", "d", "ab", "abc", "b"];
        let (permutation, skips) = optimized_continuation_permutation(&items);
        assert_eq!(permutation, vec![1, 3, 4, 5, 2, 0]);
        assert_eq!(skips, vec![2, 1, 0, 0, 1, 0]);
    }
}
