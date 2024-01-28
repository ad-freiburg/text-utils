pub mod adaptive_radix_trie;
pub mod patricia_trie;
pub mod trie;

pub trait PrefixSearch<V> {
    fn insert<K>(&mut self, key: K, value: V)
    where
        K: AsRef<[u8]>;

    fn delete<K>(&mut self, key: K) -> Option<V>
    where
        K: AsRef<[u8]>;

    fn get<K>(&self, key: K) -> Option<&V>
    where
        K: AsRef<[u8]>;

    fn contains_prefix<P>(&self, prefix: P) -> bool
    where
        P: AsRef<[u8]>;
}

pub trait ContinuationSearch<V>: PrefixSearch<V> {
    fn continuations<'a, P>(&'a self, prefix: P) -> impl Iterator<Item = (Vec<u8>, &'a V)>
    where
        P: AsRef<[u8]>,
        V: 'a;

    fn contains_continuation<P, C>(&self, prefix: P, continuation: C) -> bool
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>;

    fn contains_continuations<P, C>(&self, prefix: P, continuations: &[C]) -> Vec<bool>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        // default naive implementation, should be overridden if there is a more efficient way
        continuations
            .iter()
            .map(|c| self.contains_continuation(prefix.as_ref(), c.as_ref()))
            .collect()
    }

    fn batch_contains_continuations<P, C>(
        &self,
        prefixes: &[P],
        continuations: &[C],
    ) -> Vec<Vec<bool>>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
        Self: Sync,
    {
        // default naive implementation, should be overridden if there is a more efficient way
        prefixes
            .iter()
            .map(|p| self.contains_continuations(p, continuations))
            .collect()
    }
}
