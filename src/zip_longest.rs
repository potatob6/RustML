pub fn zip_longest<I, J>(iter1: I, iter2: J) -> ZipLongest<I::IntoIter, J::IntoIter>
where
    I: IntoIterator,
    J: IntoIterator,
{
    ZipLongest {
        iter1: iter1.into_iter(),
        iter2: iter2.into_iter(),
    }
}

pub struct ZipLongest<A, B> {
    iter1: A,
    iter2: B,
}

impl<A, B> Iterator for ZipLongest<A, B>
where
    A: Iterator,
    B: Iterator,
{
    type Item = (Option<A::Item>, Option<B::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.iter1.next();
        let b = self.iter2.next();

        if a.is_none() && b.is_none() {
            None
        } else {
            Some((a, b))
        }
    }
}
