use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

use crate::prefix::PrefixTreeSearch;

#[derive(Serialize, Deserialize)]
pub struct PrefixVec<V> {
    pub data: Vec<(Vec<u8>, V)>,
}

pub(crate) enum FindResult {
    Found(usize, usize),
    NotFound(usize),
}

impl<V> Default for PrefixVec<V> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl<V> PrefixVec<V> {
    #[inline]
    fn binary_search(
        &self,
        key: &u8,
        depth: usize,
        mut left: usize,
        mut right: usize,
    ) -> anyhow::Result<usize, usize> {
        // perform binary search over bytes at given depth
        let mut size = right - left;
        while left < right {
            let mid = left + size / 2;

            let cur = &self.data[mid].0;

            let cmp = if depth >= cur.len() {
                Ordering::Greater
            } else {
                cur[depth].cmp(key)
            };

            if cmp == Ordering::Less {
                left = mid + 1;
            } else if cmp == Ordering::Greater {
                right = mid;
            } else {
                return Ok(mid);
            }
            size = right - left;
        }
        Err(left)
    }

    #[inline]
    fn range_search(
        &self,
        key: &u8,
        depth: usize,
        left: usize,
        right: usize,
    ) -> Option<(usize, usize)> {
        let idx = match self.binary_search(key, depth, left, right) {
            Err(_) => return None,
            Ok(idx) => idx,
        };

        let right_bound = |right_idx: usize| -> bool {
            right_idx < right
                && depth < self.data[right_idx].0.len()
                && self.data[right_idx].0[depth] <= *key
        };

        // exponential search to overshoot right bound
        let mut right_idx = idx;
        let mut i = 0;
        while right_bound(right_idx) {
            right_idx += 2usize.pow(i);
            i += 1;
        }

        // search linearly from the previous exponential search position
        // to find right bound
        right_idx = idx + 2usize.pow(i - 1);
        while right_bound(right_idx) {
            right_idx += 1;
        }

        // now do the same for the left bound, a little bit
        // different here since left is inclusive
        let left_bound = |left_idx: usize| -> bool {
            left_idx > left
                && depth < self.data[left_idx - 1].0.len()
                && self.data[left_idx - 1].0[depth] >= *key
        };
        let mut left_idx = idx;
        i = 0;
        while left_bound(left_idx) {
            left_idx = left_idx.saturating_sub(2usize.pow(i));
            i += 1;
        }

        if i > 0 {
            left_idx = idx.saturating_sub(2usize.pow(i - 1));
            while left_bound(left_idx) {
                left_idx -= 1;
            }
        }
        Some((left_idx, right_idx))
    }

    #[inline]
    pub(crate) fn find_range(
        &self,
        key: &[u8],
        mut left: usize,
        mut right: usize,
        start_depth: usize,
    ) -> FindResult {
        for (depth, k) in key.iter().enumerate() {
            let Some((new_left, new_right)) = self.range_search(k, start_depth + depth, left, right) else {
                return FindResult::NotFound(depth);
            };
            left = new_left;
            right = new_right;
        }
        FindResult::Found(left, right)
    }
}

impl<V> PrefixTreeSearch<V> for PrefixVec<V> {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn insert(&mut self, key: &[u8], value: V) {
        match self
            .data
            .binary_search_by(|(prefix, _)| prefix.as_slice().cmp(key))
        {
            Ok(idx) => self.data[idx].1 = value,
            Err(idx) => self.data.insert(idx, (key.to_vec(), value)),
        };
    }

    fn get(&self, prefix: &[u8]) -> Option<&V> {
        match self.find_range(prefix, 0, self.size(), 0) {
            FindResult::Found(left, _) => {
                if self.data[left].0.len() != prefix.len() {
                    None
                } else {
                    Some(&self.data[left].1)
                }
            }
            _ => None,
        }
    }

    fn get_mut(&mut self, prefix: &[u8]) -> Option<&mut V> {
        match self.find_range(prefix, 0, self.size(), 0) {
            FindResult::Found(left, _) => {
                if self.data[left].0.len() != prefix.len() {
                    None
                } else {
                    Some(&mut self.data[left].1)
                }
            }
            _ => None,
        }
    }

    fn contains(&self, prefix: &[u8]) -> bool {
        matches!(
            self.find_range(prefix, 0, self.size(), 0),
            FindResult::Found(..)
        )
    }

    fn get_continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (Vec<u8>, &V)> + '_> {
        match self.find_range(prefix, 0, self.size(), 0) {
            FindResult::NotFound(..) => Box::new(std::iter::empty()),
            FindResult::Found(left, right) => {
                Box::new(self.data[left..right].iter().map(|(k, v)| (k.clone(), v)))
            }
        }
    }
}

impl<V> FromIterator<(Vec<u8>, V)> for PrefixVec<V> {
    fn from_iter<T: IntoIterator<Item = (Vec<u8>, V)>>(iter: T) -> Self {
        let mut pfx = Self::default();
        for (key, value) in iter {
            if key.is_empty() {
                continue;
            }
            pfx.data.push((key, value));
        }
        // sort by prefix
        pfx.data.sort_by(|(a, _), (b, _)| a.cmp(b));
        // mark duplicate prefixes
        let mark: Vec<_> = pfx
            .data
            .iter()
            .enumerate()
            .map(|(i, (key, _))| {
                if i == 0 {
                    false
                } else {
                    key == &pfx.data[i - 1].0
                }
            })
            .collect();
        // filter out duplicate prefixes
        pfx.data = pfx
            .data
            .into_iter()
            .zip(mark)
            .filter_map(|((k, v), mark)| if mark { None } else { Some((k, v)) })
            .collect();
        pfx
    }
}

#[cfg(test)]
mod tests {
    use crate::prefix::PrefixTreeSearch;

    use super::PrefixVec;

    #[test]
    fn test_prefix_vec() {
        let pfx = [
            (b"a".to_vec(), 1),
            (b"ab".to_vec(), 2),
            (b"abc".to_vec(), 3),
            (b"abcd".to_vec(), 4),
            (b"abcde".to_vec(), 5),
            (b"abd".to_vec(), 6),
            (b"bde".to_vec(), 7),
        ]
        .into_iter()
        .collect::<PrefixVec<_>>();

        assert_eq!(pfx.get(b"a"), Some(&1));
        assert!(pfx.contains(b"a"));
        assert!(!pfx.contains(b"bbd"));
        assert_eq!(pfx.get(b"ab"), Some(&2));
        assert_eq!(pfx.get(b"abf"), None);
        assert!(!pfx.contains(b"abf"));
        assert_eq!(pfx.get(b"abd"), Some(&6));
    }
}
