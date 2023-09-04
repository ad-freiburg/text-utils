use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

use crate::{prefix::PrefixTreeSearch, prefix_tree::Node};

#[derive(Serialize, Deserialize)]
pub struct PrefixVec<V> {
    pub data: Vec<(Vec<u8>, V)>,
    range_memo: Option<(Node<(usize, usize)>, usize)>,
}

pub(crate) enum FindResult {
    Found(usize, usize),
    NotFound(usize),
}

impl<V> Default for PrefixVec<V> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            range_memo: None,
        }
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
        let skip = if start_depth == 0 && !key.is_empty() {
            let mut skip = 0;
            if let Some((memo, max_depth)) = &self.range_memo {
                let max_length = key.len().min(*max_depth);
                if let Some(range) = memo.get(&key[..max_length]) {
                    left = range.0;
                    right = range.1;
                    if *max_depth >= key.len() {
                        return FindResult::Found(left, right);
                    }
                    skip = *max_depth;
                } else {
                    return FindResult::NotFound(max_length);
                }
            }
            skip
        } else {
            0
        };
        for (depth, k) in key.iter().enumerate().skip(skip) {
            let Some((new_left, new_right)) = self.range_search(k, start_depth + depth, left, right) else {
                return FindResult::NotFound(depth);
            };
            left = new_left;
            right = new_right;
        }
        FindResult::Found(left, right)
    }

    fn get_memo(
        &self,
        mut memo: Node<(usize, usize)>,
        pfx: Vec<u8>,
        d: usize,
        max_d: usize,
    ) -> Node<(usize, usize)> {
        if d > max_d {
            return memo;
        }
        let Some(&(left, right)) = memo.get(&pfx) else {
            return memo;
        };
        for k in 0..=255 {
            if let Some(range) = self.range_search(&k, d, left, right) {
                let mut pfx_k = pfx.clone();
                pfx_k.push(k);
                memo.insert(&pfx_k, range);
                memo = self.get_memo(memo, pfx_k, d + 1, max_d);
            };
        }
        memo
    }

    pub fn compute_memo(&mut self, max_depth: usize) {
        self.range_memo.take();
        let mut memo = Node::default();
        memo.insert(&[], (0, self.size()));
        memo = self.get_memo(memo, vec![], 0, max_depth);
        self.range_memo = Some((memo, max_depth));
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
        let mut pfx: PrefixVec<_> = [
            (b"a".to_vec(), 1),
            (b"ab".to_vec(), 2),
            (b"abc".to_vec(), 3),
            (b"abcd".to_vec(), 4),
            (b"abcde".to_vec(), 5),
            (b"abd".to_vec(), 6),
            (b"bde".to_vec(), 7),
        ]
        .into_iter()
        .collect();

        assert_eq!(pfx.get(b"a"), Some(&1));
        assert!(pfx.contains(b"a"));
        assert!(!pfx.contains(b"bbd"));
        assert_eq!(pfx.get(b"ab"), Some(&2));
        assert_eq!(pfx.get(b"abf"), None);
        assert!(!pfx.contains(b"abf"));
        assert_eq!(pfx.get(b"abd"), Some(&6));

        pfx.compute_memo(2);

        assert_eq!(pfx.get(b"a"), Some(&1));
        assert!(pfx.contains(b"a"));
        assert!(!pfx.contains(b"bbd"));
        assert_eq!(pfx.get(b"ab"), Some(&2));
        assert_eq!(pfx.get(b"abf"), None);
        assert!(!pfx.contains(b"abf"));
        assert_eq!(pfx.get(b"abd"), Some(&6));
    }
}
