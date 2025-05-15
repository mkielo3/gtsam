import numpy as np
import manifpy


lieplusplus_jac = np.array([[ 1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -9.99997e-03, -5.15856e-06, -1.99352e-05,  4.99998e-05,  3.43405e-08,  1.32914e-07,  1.95404e-10, -8.11886e-07,  1.44360e-07,  1.25006e-04],
		[ 0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.27809e-06, -9.99986e-03, -4.49980e-05, -3.52370e-08,  4.99990e-05,  2.99981e-07,  8.09647e-07, -2.18288e-09,  4.18537e-07, -4.45647e-05],
		[ 0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.99039e-05,  4.50119e-05, -9.99984e-03, -1.32679e-07, -3.00085e-07,  4.99988e-05, -1.49507e-07, -4.16229e-07, -3.25675e-09, -2.43252e-04],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -9.99997e-03, -5.15856e-06, -1.99352e-05, -5.29197e-08,  2.43515e-04, -4.31042e-05,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.27809e-06, -9.99986e-03, -4.49980e-05, -2.42919e-04,  5.82925e-07, -1.25650e-04,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.99039e-05,  4.50119e-05, -9.99984e-03,  4.44764e-05,  1.25034e-04,  8.68459e-07,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -9.99997e-03, -5.15856e-06, -1.99352e-05,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.27809e-06, -9.99986e-03, -4.49980e-05,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.99039e-05,  4.50119e-05, -9.99984e-03,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -1.00000e-02],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])

gtsamcpp_jac = np.array([[ 1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -9.99997e-03, -5.15856e-06, -1.99352e-05,  4.99998e-05,  3.43405e-08,  1.32914e-07,  1.95404e-10, -8.11886e-07,  1.44360e-07,  1.25006e-04],
		[ 0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.27809e-06, -9.99986e-03, -4.49980e-05, -3.52370e-08,  4.99990e-05,  2.99981e-07,  8.09647e-07, -2.18288e-09,  4.18537e-07, -4.45647e-05],
		[ 0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.99039e-05,  4.50119e-05, -9.99984e-03, -1.32679e-07, -3.00085e-07,  4.99988e-05, -1.49507e-07, -4.16229e-07, -3.25675e-09, -2.43252e-04],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -9.99997e-03, -5.15856e-06, -1.99352e-05, -5.29197e-08,  2.43515e-04, -4.31042e-05,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.27809e-06, -9.99986e-03, -4.49980e-05, -2.42919e-04,  5.82925e-07, -1.25650e-04,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.99039e-05,  4.50119e-05, -9.99984e-03,  4.44764e-05,  1.25034e-04,  8.68459e-07,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -9.99997e-03, -5.15856e-06, -1.99352e-05,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.27809e-06, -9.99986e-03, -4.49980e-05,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.99039e-05,  4.50119e-05, -9.99984e-03,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -1.00000e-02],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])

gtsamcpp_cov = np.array([[ 2.50001e-13, -4.35704e-19,  4.67907e-19, -5.00001e-11, -8.56647e-15, -3.33398e-14, -6.43729e-19,  8.11226e-15, -1.48012e-15, -1.25006e-24,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-4.35704e-19,  2.50001e-13, -1.79252e-19,  8.82802e-15, -4.99999e-11, -7.49550e-14, -8.10478e-15,  7.26861e-18, -4.16928e-15,  4.45647e-25,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 4.67907e-19, -1.79252e-19,  2.50000e-13,  3.30590e-14,  7.50622e-14, -4.99997e-11,  1.49728e-15,  4.16159e-15,  1.08560e-17,  2.43252e-24,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-5.00001e-11,  8.82802e-15,  3.30590e-14,  1.00000e-08, -2.43122e-14,  3.82364e-14,  1.32299e-16, -2.43318e-12,  4.41995e-13,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-8.56647e-15, -4.99999e-11,  7.50622e-14, -2.43122e-14,  1.00000e-08, -1.43054e-14,  2.43169e-12, -1.45732e-15,  1.25167e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-3.33398e-14, -7.49550e-14, -4.99997e-11,  3.82364e-14, -1.43054e-14,  9.99994e-09, -4.45425e-13, -1.25013e-12, -2.17115e-15,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-6.43729e-19, -8.10478e-15,  1.49728e-15,  1.32299e-16,  2.43169e-12, -4.45425e-13,  9.99999e-11, -2.98830e-16,  7.82845e-17,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 8.11226e-15,  7.26861e-18,  4.16159e-15, -2.43318e-12, -1.45732e-15, -1.25013e-12, -2.98830e-16,  9.99993e-11, -3.46494e-17,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-1.48012e-15, -4.16928e-15,  1.08560e-17,  4.41995e-13,  1.25167e-12, -2.17115e-15,  7.82845e-17, -3.46494e-17,  9.99992e-11,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-1.25006e-24,  4.45647e-25,  2.43252e-24,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-22,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-14,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-14,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-14,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-16,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-16,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-16,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-18]])


lieplusplus_cov = np.array([[ 2.50001e-13, -4.35704e-19,  4.67907e-19, -5.00001e-11, -8.56647e-15, -3.33398e-14, -6.43729e-19,  8.11226e-15, -1.48012e-15, -1.25006e-24,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-4.35704e-19,  2.50001e-13, -1.79252e-19,  8.82802e-15, -4.99999e-11, -7.49550e-14, -8.10478e-15,  7.26861e-18, -4.16928e-15,  4.45647e-25,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 4.67907e-19, -1.79252e-19,  2.50000e-13,  3.30590e-14,  7.50622e-14, -4.99997e-11,  1.49728e-15,  4.16159e-15,  1.08560e-17,  2.43252e-24,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-5.00001e-11,  8.82802e-15,  3.30590e-14,  1.00000e-08, -2.43122e-14,  3.82364e-14,  1.32299e-16, -2.43318e-12,  4.41995e-13,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-8.56647e-15, -4.99999e-11,  7.50622e-14, -2.43122e-14,  1.00000e-08, -1.43054e-14,  2.43169e-12, -1.45732e-15,  1.25167e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-3.33398e-14, -7.49550e-14, -4.99997e-11,  3.82364e-14, -1.43054e-14,  9.99994e-09, -4.45425e-13, -1.25013e-12, -2.17115e-15,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-6.43729e-19, -8.10478e-15,  1.49728e-15,  1.32299e-16,  2.43169e-12, -4.45425e-13,  9.99999e-11, -2.98830e-16,  7.82845e-17,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 8.11226e-15,  7.26861e-18,  4.16159e-15, -2.43318e-12, -1.45732e-15, -1.25013e-12, -2.98830e-16,  9.99993e-11, -3.46494e-17,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-1.48012e-15, -4.16928e-15,  1.08560e-17,  4.41995e-13,  1.25167e-12, -2.17115e-15,  7.82845e-17, -3.46494e-17,  9.99992e-11,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[-1.25006e-24,  4.45647e-25,  2.43252e-24,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-22,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-14,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-14,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-14,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-12,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-16,  0.00000e+00,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-16,  0.00000e+00,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-16,  0.00000e+00],
		[ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-18]])



class GalileanPreintegrationParams:
	"""Parameters for Galilean preintegration"""
	def __init__(self, gravity=np.array([0, 0, -9.81]), 
				 gyro_noise=1e-4, acc_noise=1e-3, 
				 virtual_vel_noise=1e-8, virtual_time_noise=1e-10,
				 gyro_bias_noise=1e-6, acc_bias_noise=1e-5,
				 virtual_vel_bias_noise=1e-7, virtual_time_bias_noise=1e-8):
		self.n_gravity = gravity
		self.gyroscope_covariance = np.eye(3) * (gyro_noise * gyro_noise)
		self.accelerometer_covariance = np.eye(3) * (acc_noise * acc_noise)
		self.virtual_vel_covariance = np.eye(3) * (virtual_vel_noise * virtual_vel_noise)
		self.virtual_time_scale_covariance = virtual_time_noise * virtual_time_noise
		self.bias_omega_covariance = np.eye(3) * (gyro_bias_noise * gyro_bias_noise)
		self.bias_acc_covariance = np.eye(3) * (acc_bias_noise * acc_bias_noise)
		self.bias_virtual_vel_covariance = np.eye(3) * (virtual_vel_bias_noise * virtual_vel_bias_noise)
		self.bias_virtual_time_covariance = virtual_time_bias_noise * virtual_time_bias_noise

class Bias:
	"""IMU bias class"""
	def __init__(self, gyroscope=np.zeros(3), accelerometer=np.zeros(3)):
		self.gyro = gyroscope
		self.acc = accelerometer
		
	def gyroscope(self):
		return self.gyro
	
	def accelerometer(self):
		return self.acc
	
	def vector(self):
		return np.concatenate([self.acc, self.gyro])

class PreintegratedGalileanMeasurements:
	"""
	Class to handle preintegration of IMU measurements using the Galilean group
	
	The tangent space ordering in Manif vs GTSAM:
	- Manif:  [w, v, p, s] = [gyro(0-2), acc(3-5), position(6-8), time(9)]
	- GTSAM: [rho, nu, theta, t] = [position(0-2), velocity(3-5), rotation(6-8), time(9)]
	"""
	
	def __init__(self, params, bias_hat=None):
		self.params = params
		self.bias_hat = bias_hat if bias_hat is not None else Bias()
		self.resetIntegration()
		
	def resetIntegration(self):
		"""Reset integration to initial state"""
		self.delta_upsilon = manifpy.SGal3()
		self.delta_upsilon.setIdentity()
		self.delta_tij = 0.0
		self.preint_meas_cov = np.zeros((20, 20))
		self.preint_bias_jacobian = np.eye(20)
		
	def mapBias6ToTangent10(self, bias6D):
		"""
		Convert 6D bias [acc(0-2), gyro(3-5)] to 10D bias in GTSAM ordering
		[rho(0-2), nu(3-5), theta(6-8), t(9)]
		"""
		result = np.zeros(10)
		result[6:9] = bias6D[0:3]  # gyro bias to theta (rotation) - indices 6-8
		result[3:6] = bias6D[3:6]  # acc bias to nu (velocity) - indices 3-5
		# Position bias and time bias remain zero
		return result

	
	def convert_to_gtsam_ordering(self, manif_vec):
		"""Convert 10D vector from Manif to GTSAM ordering"""
		gtsam_vec = np.zeros_like(manif_vec)
		gtsam_vec[0:3] = manif_vec[6:9]  # position
		gtsam_vec[3:6] = manif_vec[3:6]  # velocity
		gtsam_vec[6:9] = manif_vec[0:3]  # rotation
		gtsam_vec[9] = manif_vec[9]	  # time
		return gtsam_vec
	
	def convert_to_manif_ordering(self, gtsam_vec):
		"""Convert 10D vector from GTSAM to Manif ordering"""
		manif_vec = np.zeros_like(gtsam_vec)
		manif_vec[0:3] = gtsam_vec[6:9]  # rotation
		manif_vec[3:6] = gtsam_vec[3:6]  # velocity
		manif_vec[6:9] = gtsam_vec[0:3]  # position
		manif_vec[9] = gtsam_vec[9]	  # time
		return manif_vec

	def mapMeasurement10ToTangent10(self, measurement10D):
		"""
		Convert measurement from Manif ordering [gyro, acc, virtual_vel, virtual_time]
		to GTSAM ordering [rho, nu, theta, t] = [position, velocity, rotation, time]
		"""
		tangent = np.zeros(10)
		# Position (rho) - from virtual_vel in Manif
		tangent[0:3] = measurement10D[6:9]
		# Velocity (nu) - from acc in Manif
		tangent[3:6] = measurement10D[3:6]
		# Rotation (theta) - from gyro in Manif
		tangent[6:9] = measurement10D[0:3]
		# Time (t) - from virtual_time in Manif
		tangent[9] = measurement10D[9]
		
		return tangent


	def convert_matrix_from_gtsam_to_manif_ordering(self, gtsam_mat):
		"""Convert a matrix from GTSAM to Manif ordering"""
		n = gtsam_mat.shape[0]
		manif_mat = np.zeros_like(gtsam_mat)
		
		# Create the permutation for the first 10 elements (inverse of previous permutation)
		perm10 = [6, 7, 8, 3, 4, 5, 0, 1, 2, 9]
		
		# Create the inverse permutation
		inv_perm10 = [0] * 10
		for i, p in enumerate(perm10):
			inv_perm10[p] = i
		
		# For 20x20 matrix, extend the permutation
		if n == 20:
			inv_perm = inv_perm10 + [i+10 for i in inv_perm10]
		else:
			inv_perm = inv_perm10
		
		# Apply permutation to rows and columns
		for i in range(n):
			for j in range(n):
				manif_mat[i, j] = gtsam_mat[inv_perm[i], inv_perm[j]]
				
		return manif_mat
		
	def integrateMeasurement(self, measuredAcc, measuredOmega, dt):
		"""Integrate a single IMU measurement using the Galilean propagation."""
		
		if dt <= 0:
			print("WARNING: dt <= 0 in integrateMeasurement. Skipping integration.")
			return

		# Create input with virtual components set to zero
		print(f"=== PYTHON INTEGRATION STEP ===")
		print(f"Input acc:  {measuredAcc[0]} {measuredAcc[1]} {measuredAcc[2]}")
		print(f"Input gyro: {measuredOmega[0]} {measuredOmega[1]} {measuredOmega[2]}")
		print(f"Input dt: {dt}")
		
		# Create measurement vector
		w = np.zeros(10)
		w[0:3] = measuredOmega
		w[3:6] = measuredAcc
		w[9] = 1.0  # virtual time scale = 1
		print(f"Full measurement w: {' '.join(f'{x:.6f}' for x in w)}")
		
		# Remove bias from measurement
		bias_10D = self.mapBias6ToTangent10(self.bias_hat.vector())
		print(f"Bias vector (10D): {' '.join(f'{x:.6f}' for x in bias_10D)}")
		
		w_unbiased = w - bias_10D
		print(f"Unbiased measurement: {' '.join(f'{x:.6f}' for x in w_unbiased)}")
		
		# Convert from Manif ordering to GTSAM tangent space
		tangent_arg = self.mapMeasurement10ToTangent10(w_unbiased)
		print(f"Tangent coeffs (after reordering): {' '.join(f'{x:.6f}' for x in tangent_arg)}")
		
		# Scale by dt
		tangent_scaled = tangent_arg * dt
		print(f"Tangent coeffs (after dt scaling): {' '.join(f'{x:.6f}' for x in tangent_scaled)}")
		
		# Compute left Jacobian
		tangent_obj = manifpy.SGal3Tangent()
		for i in range(10):
			tangent_obj.coeffs()[i] = tangent_scaled[i]
		Lj = np.array(tangent_obj.ljac())
		print(f"Left Jacobian shape: {Lj.shape[0]}x{Lj.shape[1]}")
		print(f"Left Jacobian first few elements: {' '.join(f'{x:.6f}' for x in Lj[0, :])}")
		
		# Get adjoint map from current delta_upsilon
		# Changed AdjointMap() to adj()
		Adj_Upsilon = np.array(self.delta_upsilon.adj())
		print(f"Adjoint map shape: {Adj_Upsilon.shape[0]}x{Adj_Upsilon.shape[1]}")
		print(f"Adjoint trace: {np.trace(Adj_Upsilon)}")
		
		# Compute K matrix for covariance update
		K = (Adj_Upsilon @ Lj) * dt
		print(f"K matrix shape: {K.shape[0]}x{K.shape[1]}")
		print(f"K matrix trace: {np.trace(K)}")
		
		# Print current delta_upsilon before update
		print(f"Delta upsilon before:")
		print(f"R: {np.array(self.delta_upsilon.rotation())}")
		print(f"r: {' '.join(f'{x:.6f}' for x in self.delta_upsilon.translation())}")
		print(f"v: {' '.join(f'{x:.6f}' for x in self.delta_upsilon.linearVelocity())}")
		print(f"t: {self.delta_upsilon.t()}")
		
		# Update mean using the exponential map
		exp_result = tangent_obj.exp()
		self.delta_upsilon = self.delta_upsilon.compose(exp_result)
		self.delta_tij += dt
		
		# Print updated delta_upsilon
		print(f"Delta upsilon after:")
		print(f"R: {np.array(self.delta_upsilon.rotation())}")
		print(f"r: {' '.join(f'{x:.6f}' for x in self.delta_upsilon.translation())}")
		print(f"v: {' '.join(f'{x:.6f}' for x in self.delta_upsilon.linearVelocity())}")
		print(f"t: {self.delta_upsilon.t()}")
		print(f"Delta t after: {self.delta_tij}")
		
		# Print the result of Expmap
		print(f"Exp result:")
		print(f"R: {np.array(exp_result.rotation())}")
		print(f"r: {' '.join(f'{x:.6f}' for x in exp_result.translation())}")
		print(f"v: {' '.join(f'{x:.6f}' for x in exp_result.linearVelocity())}")
		print(f"t: {exp_result.t()}")
		
		# Get adjoint map of the exponential result
		# Changed AdjointMap() to adj()
		exp_adj = np.array(exp_result.adj())
		print(f"Exp adjoint shape: {exp_adj.shape[0]}x{exp_adj.shape[1]}")
		print(f"Exp adjoint trace: {np.trace(exp_adj)}")
		
		# Propagate covariance and Jacobians
		A = np.eye(20)
		A[0:10, 10:20] = Lj * dt
		A[10:20, 10:20] = exp_adj
		print(f"A matrix shape: {A.shape[0]}x{A.shape[1]}")
		print(f"A matrix diagonal: {' '.join(f'{x:.6f}' for x in np.diag(A))}")
		
		B = np.zeros((20, 20))
		B[0:10, 0:10] = -K
		# Changed AdjointMap() to adj()
		B[10:20, 10:20] = np.array(self.delta_upsilon.adj()) * dt
		print(f"B matrix shape: {B.shape[0]}x{B.shape[1]}")
		print(f"B matrix trace: {np.trace(B)}")
		
		# Create noise covariance Q_d
		params = self.params
		Q_d = np.zeros((20, 20))
		# Fill Q_d with noise covariances
		Q_d[0:3, 0:3] = params.virtual_vel_covariance / dt  # rho (position)
		Q_d[3:6, 3:6] = params.accelerometer_covariance / dt   # nu (velocity)
		Q_d[6:9, 6:9] = params.gyroscope_covariance / dt      # theta (rotation)
		Q_d[9, 9] = params.virtual_time_scale_covariance / dt  # t (time)
		
		Q_d[10:13, 10:13] = params.bias_omega_covariance / dt  # gyro bias
		Q_d[13:16, 13:16] = params.bias_acc_covariance / dt    # acc bias
		Q_d[16:19, 16:19] = params.bias_virtual_vel_covariance / dt  # virtual vel bias
		Q_d[19, 19] = params.bias_virtual_time_covariance / dt   # virtual time bias
		print(f"Q_d matrix shape: {Q_d.shape[0]}x{Q_d.shape[1]}")
		print(f"Q_d trace: {np.trace(Q_d)}")
		
		# Save old matrices for comparison
		old_cov = self.preint_meas_cov.copy()
		old_jac = self.preint_bias_jacobian.copy()
		
		# Update covariance
		self.preint_meas_cov = A @ self.preint_meas_cov @ A.T + B @ Q_d @ B.T
		
		# Compute and print change norms
		cov_change = np.linalg.norm(self.preint_meas_cov - old_cov)
		print(f"Covariance matrix change norm: {cov_change}")
		
		# Update bias Jacobian
		Phi_b = np.eye(20, dtype=np.float128)
		Phi_b[0:10, 10:20] = -1.0 * K
		self.preint_bias_jacobian = Phi_b @ self.preint_bias_jacobian
		
		jac_change = np.linalg.norm(self.preint_bias_jacobian - old_jac)
		print(f"Jacobian matrix change norm: {jac_change}")
		
		# Ensure numerical stability by enforcing symmetry for covariance
		self.preint_meas_cov = 0.5 * (self.preint_meas_cov + self.preint_meas_cov.T)
		
		# Print some key submatrices for comparison
		print(f"Covariance block [0:3, 0:3]:")
		for i in range(3):
			print(' '.join(f'{x:.6e}' for x in self.preint_meas_cov[i, 0:3]))
		
		print(f"Jacobian block [0:3, 0:3]:")
		for i in range(3):
			print(' '.join(f'{x:.6f}' for x in self.preint_bias_jacobian[i, 0:3]))
		
		print(f"=== END PYTHON INTEGRATION ===")

		
	def deltaRij(self):
		"""Get rotation component"""
		return np.array(self.delta_upsilon.rotation())
	
	def deltaPij(self):
		"""Get translation component"""
		return np.array(self.delta_upsilon.translation())
	
	def deltaVij(self):
		"""Get velocity component"""
		return np.array(self.delta_upsilon.linearVelocity())
	
	def deltaTij(self):
		"""Get time component"""
		return self.delta_tij
		
	def uncertaintyCovariance(self):
		"""Get uncertainty covariance matrix in GTSAM ordering"""
		return self.preint_meas_cov
	
	def biasJacobian(self):
		"""Get bias Jacobian matrix in GTSAM ordering"""
		return self.preint_bias_jacobian

def flattenAndSort(m):
	"""Flatten and sort matrix elements for comparison"""
	v = m.flatten()
	v.sort()
	return v

def assert_equal(expected, actual, tol):
	"""Check if matrices are equal within tolerance"""
	diff = np.abs(expected - actual)
	max_diff = np.max(diff)
	result = max_diff <= tol
	return result, max_diff



from copy import deepcopy

def run_single_step_debug_test():
	"""Run the SingleStepDebug test with exact expected values"""
	print("======= Running SingleStepDebug Test =======")
	
	# Create parameters with the same values as in the C++ test
	p = GalileanPreintegrationParams(
		gravity=np.array([0, 0, -9.81]), 
		gyro_noise=1e-4, acc_noise=1e-3, 
		virtual_vel_noise=1e-8, virtual_time_noise=1e-10,
		gyro_bias_noise=1e-6, acc_bias_noise=1e-5,
		virtual_vel_bias_noise=1e-7, virtual_time_bias_noise=1e-8
	)
	
	# Create IMU preintegrator
	pim = PreintegratedGalileanMeasurements(p)
	pim.resetIntegration()
	
	# Use the exact same measurement from the C++ test
	acc = np.array([-2.5068965533954604, 0.87582680732853024, 4.8643795488708914])
	gyro = np.array([-0.90010662229654148, 0.39839438991809351, -0.10436735729977142])
	dt = 0.01
	
	# Integrate once
	pim.integrateMeasurement(acc, gyro, dt)
	
	# Print basic integration results
	print("\nBasic Integration Results:")
	print("Delta Time:", pim.deltaTij())
	
	pim_converter = PreintegratedGalileanMeasurements(p)  # Use a fresh instance for conversion
	expectedCov_gtsam = deepcopy(gtsamcpp_cov)
	expectedJxi_gtsam = deepcopy(gtsamcpp_jac)
	# expectedCov_gtsam = deepcopy(lieplusplus_cov)
	# expectedJxi_gtsam = deepcopy(lieplusplus_jac)

	# Get actual matrices
	actualCov = pim.uncertaintyCovariance()
	actualJxi = pim.biasJacobian()
	
	# Compare matrices directly
	print("\nComparing Covariance Matrices:")
	cov_equal, cov_max_diff = assert_equal(expectedCov_gtsam, actualCov, 1e-10)
	print(f"Direct comparison result: {'PASSED' if cov_equal else 'FAILED'}")
	print(f"Max difference: {cov_max_diff:.2e}")
	
	print("\nComparing Jacobian Matrices:")
	jxi_equal, jxi_max_diff = assert_equal(expectedJxi_gtsam, actualJxi, 1e-10)
	print(f"Direct comparison result: {'PASSED' if jxi_equal else 'FAILED'}")
	print(f"Max difference: {jxi_max_diff:.2e}")
	
	# Also compare sorted values as in the C++ test
	print("\nComparing Sorted Flattened Matrices:")
	expectedSorted = flattenAndSort(expectedCov_gtsam)
	actualSorted = flattenAndSort(pim.preint_meas_cov)
	
	sorted_equal, sorted_max_diff = assert_equal(expectedSorted, actualSorted, 1e-10)
	print(f"Sorted covariance comparison: {'PASSED' if sorted_equal else 'FAILED'}")
	print(f"Max difference: {sorted_max_diff:.2e}")
	
	expectedSorted2 = flattenAndSort(expectedJxi_gtsam)
	actualSorted2 = flattenAndSort(pim.preint_bias_jacobian)
	
	sorted2_equal, sorted2_max_diff = assert_equal(expectedSorted2, actualSorted2, 1e-10)
	print(f"Sorted jacobian comparison: {'PASSED' if sorted2_equal else 'FAILED'}")
	print(f"Max difference: {sorted2_max_diff:.2e}")
	
	# Overall test result
	overall_result = cov_equal and jxi_equal and sorted_equal and sorted2_equal
	print("\n======= SingleStepDebug Test Result =======")
	print(f"Overall Result: {'PASSED' if overall_result else 'FAILED'}")
	print("===========================================")
	
	return expectedJxi_gtsam, actualJxi, expectedCov_gtsam, actualCov


expectedJxi_gtsam, actualJxi, expectedCov_gtsam, actualCov = run_single_step_debug_test()