# iterative_closest_point_2d

inspired by http://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

#usage
Call like this,
```python
ret = icp(d1, d2)
```
.
`d1, d2` are numpy array of 2d points.

The return value `ret` is the convert matrix with 2 rows and 3 coloums.

`icp` estimates rotation, moving, scaling(each x and y Separately) convertion.
