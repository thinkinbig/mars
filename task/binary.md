# Binary Search Notation

Sorry that I was previously reluctant to write down the notation due to the fact that it's a bit complicated and afar from the topic. But several questions must be answered to prove the algorithm correct.

## The Algorithm

``` python
    def knot_index(self, v):
        if self.knots[0] > v or self.knots[-1] < v:
            raise ValueError("knot value out of range")
        # binary search right most index
        l, r = 0, len(self.knots) - 1
        while l < r:
            m = (l + r) // 2
            if self.knots[m] > v:
                r = m
            else:
                l = m + 1
        return r - 1
```

__Question 1__ What if the first search hit the target value?

__Answer__ Say, we have the `target` value `0`, and the `array` `[0, 0, 0, 1, 2]`, the searched target index should be `2`, which is the first search hit.

### loop
| step | l | r | m|
| --- | --- | ---| --- |
|step 0 | l = 0 | r = 4 | m = 2|
|step 1 | l = 3 | r = 4 | m = 3|
|step 2 | l = 3 | r = 3 | m = 3|
end loop

After loop, return `r - 1 = 2`.


__Question 2__ More general, what if you hit the target before `r-l` shrinks to `1`? Is it still possible to output l as the left end of the interval containing `t`?

__Answer__ The algorithm aims to output the right-most index of the same value `v`, if existing. We observed that for all index smaller than the target index, they preserve the property that
$$
\forall i \leq index(v): array[i] \leq v
$$

Which implies that if the target is to be reached, it must be the left cursor firstly updated to it(or surpass the searched index by 1, depends on the case).

In both case however, the right cursor will be updated to the position `index(v) + 1`.

- _Case 1_  left cursor surpass the index by 1.
  Since
  $$ v < array[l] \leq array[r] $$
  right cursor will be ultimately updated to light cursor, which is the `index(v) + 1`, see example in __Question 1__
- _Case 2_ left cursor reached `index(v)`. The same reason goes as above
  $$ v = array[l] \leq array[r] $$
  On the last step, when `r - l = 1`, the left cursor will be updated to `index(v) + 1`. which is exactly case 1.

Above all, We have proven that `r - 1 = index(v)`.