package tests

import bfin "../wotan/behavioral_finance"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Newswatchers and momentum traders: Hong & Stein (1999), behavioral_finance/hong_stein.odin
//   hong_stein_reference_test: newswatchers only (phi = 0: linear diffusion, MA
//     autocovariances (z - k) sigma^2 / z^2, pricing error sd); the AR radius for
//     j = 1 (sqrt(phi)) and Lemma A.1's |phi| < 1; autocovariances = statsmodels
//     arma_acovf off equilibrium; Tables AI - AIII (18 parameter sets: horizon j, risk
//     tolerance gamma, diffusion z): phi, sd of returns, sd of pricing errors with and
//     without momentum traders, cumulative impulse responses at lags 0 - 35 and return
//     autocorrelations at lags 1 - 25 to the printed decimals; Lemma 1 and
//     Proposition 1 (overshooting, peak at lag j when j >= z - 1) on all 18;
//     Proposition 2 with risk-neutral momentum traders
//   hong_stein_test: three simulated markets (2000 stocks, 1200 months, monthly news
//     sigma 0.5): Hong-Stein at the base equilibrium, newswatchers only (phi = 0) and
//     rational (random walk). Pooled return autocorrelations = the model's; JT 6/6
//     momentum in both diffusion markets, DBT 36/36 reversal (losers beat winners)
//     only with momentum traders, continuation without them, nothing in the rational
//     market. The calibration has no noise besides news, so returns are very
//     persistent (lag-1 autocorrelation 0.95; JT t in the hundreds) and t-statistics
//     use 48 Newey-West lags. On 9 seeds: autocorrelations within 2.5 se, reversal t
//     6.4 - 7.8, newswatchers-only DBT t -14.6 to -20.5, rational |t| <= 2.3 (with
//     1000 stocks x 600 months the reversal t was 1.6 - 4.6)

@(private = "file")
HS_Table :: struct {
	phi, sdr, sd0, sde: [6]f64, // phi, sd(dP), sd(P - P*) at phi = 0, sd(P - P*)
	irf:                [36][6]f64, // cumulative impulse response, lags 0 - 35
	ac:                 [25][6]f64, // return autocorrelations, lags 1 - 25
}

// Tables AI - AIII (two garbled entries in the PDF text, AII lag 16 and AIII lag 17,
// restored from the scan: -0.1477, -0.1544)
@(private = "file")
HS_AI := HS_Table {
	phi = {0.555, 0.4455, 0.3262, 0.2605, 0.2263, 0.2015},
	sdr = {0.2229, 0.2322, 0.2179, 0.2028, 0.1908, 0.1833},
	sd0 = {0.9373, 0.9373, 0.9373, 0.9373, 0.9373, 0.9373},
	sde = {0.8011, 0.8438, 0.9103, 0.9365, 0.9524, 0.9604},
	irf = {
		{0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833},
		{0.2129, 0.2038, 0.1939, 0.1884, 0.1855, 0.1835},
		{0.3682, 0.3408, 0.3132, 0.2991, 0.292, 0.287},
		{0.5377, 0.4851, 0.4355, 0.4113, 0.3994, 0.3912},
		{0.6688, 0.6328, 0.5588, 0.5238, 0.5071, 0.4955},
		{0.753, 0.7819, 0.6823, 0.6365, 0.6147, 0.5999},
		{0.7969, 0.9317, 0.8059, 0.7492, 0.7225, 0.7042},
		{0.8105, 1.0446, 0.9296, 0.8618, 0.8302, 0.8086},
		{0.8287, 1.1246, 1.0533, 0.9745, 0.9379, 0.9129},
		{0.8753, 1.1825, 1.1769, 1.0872, 1.0456, 1.0173},
		{0.9602, 1.2273, 1.2734, 1.1999, 1.1533, 1.1217},
		{1.083, 1.2649, 1.3522, 1.3126, 1.261, 1.226},
		{1.1412, 1.2151, 1.3389, 1.342, 1.2854, 1.2471},
		{1.1475, 1.1263, 1.2947, 1.3279, 1.2909, 1.2513},
		{1.104, 1.0364, 1.2401, 1.2969, 1.2921, 1.2522},
		{1.0116, 0.9607, 1.182, 1.26, 1.2924, 1.2523},
		{0.9281, 0.9012, 1.1227, 1.2211, 1.2736, 1.2524},
		{0.8782, 0.8547, 1.063, 1.1817, 1.2462, 1.2524},
		{0.8747, 0.8173, 1.0032, 1.142, 1.216, 1.2524},
		{0.924, 0.8228, 0.9433, 1.1024, 1.1848, 1.2356},
		{0.9977, 0.8648, 0.8923, 1.0627, 1.1534, 1.212},
		{1.0663, 0.9235, 0.85, 1.023, 1.1219, 1.1864},
		{1.1063, 0.9834, 0.8405, 0.9833, 1.0904, 1.1603},
		{1.1012, 1.0366, 0.8518, 0.9436, 1.0589, 1.134},
		{1.0574, 1.081, 0.8733, 0.9038, 1.0274, 1.1076},
		{0.9951, 1.1175, 0.8993, 0.8859, 0.9959, 1.0813},
		{0.9382, 1.1313, 0.9271, 0.8848, 0.9644, 1.055},
		{0.9096, 1.1187, 0.9557, 0.8926, 0.9329, 1.0286},
		{0.918, 1.087, 0.9845, 0.9043, 0.9202, 1.0023},
		{0.9572, 1.0461, 1.0134, 0.9175, 0.9161, 0.9759},
		{1.0105, 1.0042, 1.0395, 0.9312, 0.9149, 0.9496},
		{1.056, 0.9658, 1.0618, 0.9451, 0.9146, 0.9401},
		{1.0766, 0.9324, 1.0722, 0.959, 0.9187, 0.9373},
		{1.0663, 0.9114, 1.0719, 0.973, 0.9259, 0.9365},
		{1.0309, 0.9076, 1.0648, 0.987, 0.9344, 0.9364},
		{0.9861, 0.9201, 1.054, 1.001, 0.9433, 0.9363},
	},
	ac  = {
		{0.863, 0.9331, 0.9458, 0.9466, 0.9465, 0.9436},
		{0.5441, 0.7888, 0.8415, 0.8504, 0.8673, 0.8634},
		{0.1706, 0.6063, 0.7048, 0.7388, 0.7813, 0.7782},
		{-0.1203, 0.4039, 0.5561, 0.622, 0.6907, 0.692},
		{-0.2115, 0.1888, 0.4031, 0.5036, 0.587, 0.6054},
		{-0.1035, -0.0293, 0.2492, 0.3846, 0.4774, 0.5182},
		{0.1284, -0.2138, 0.0963, 0.2654, 0.3658, 0.4285},
		{0.3635, -0.3233, -0.0522, 0.1464, 0.2537, 0.327},
		{0.4735, -0.3637, -0.1883, 0.0276, 0.1415, 0.2208},
		{0.4096, -0.3533, -0.3027, -0.0898, 0.03, 0.1137},
		{0.191, -0.3051, -0.3708, -0.2027, -0.0784, 0.0087},
		{-0.0958, -0.22, -0.3955, -0.2994, -0.1741, -0.0846},
		{-0.316, -0.085, -0.3589, -0.3385, -0.2162, -0.1215},
		{-0.4026, 0.0574, -0.2985, -0.3348, -0.2452, -0.1465},
		{-0.3294, 0.1696, -0.2289, -0.3088, -0.2697, -0.169},
		{-0.1297, 0.2376, -0.156, -0.2729, -0.2873, -0.1909},
		{0.1034, 0.2632, -0.0823, -0.2332, -0.2792, -0.2125},
		{0.2808, 0.2532, -0.0098, -0.1919, -0.2595, -0.233},
		{0.3387, 0.2108, 0.0582, -0.1502, -0.2355, -0.2485},
		{0.2599, 0.1318, 0.1177, -0.1083, -0.2096, -0.2402},
		{0.0869, 0.0331, 0.1594, -0.0663, -0.1803, -0.2224},
		{-0.1076, -0.0608, 0.181, -0.0245, -0.1488, -0.2016},
		{-0.2477, -0.1329, 0.1762, 0.017, -0.1165, -0.1801},
		{-0.2817, -0.1765, 0.1549, 0.0573, -0.0838, -0.1583},
		{-0.2046, -0.1914, 0.1252, 0.0929, -0.051, -0.1363},
	},
}

@(private = "file")
HS_AII := HS_Table {
	phi = {0.1316, 0.1453, 0.1625, 0.1848, 0.2152, 0.2605},
	sdr = {0.1662, 0.1691, 0.1731, 0.1786, 0.1872, 0.2028},
	sd0 = {0.9373, 0.9373, 0.9373, 0.9373, 0.9373, 0.9373},
	sde = {0.907, 0.8998, 0.8999, 0.9021, 0.9102, 0.9365},
	irf = {
		{0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833},
		{0.1776, 0.1788, 0.1802, 0.1821, 0.1846, 0.1884},
		{0.2734, 0.276, 0.2793, 0.2836, 0.2897, 0.2991},
		{0.3693, 0.3734, 0.3787, 0.3857, 0.3957, 0.4113},
		{0.4652, 0.4709, 0.4782, 0.4879, 0.5018, 0.5238},
		{0.5612, 0.5684, 0.5777, 0.5902, 0.608, 0.6365},
		{0.6572, 0.6659, 0.6772, 0.6924, 0.7142, 0.7492},
		{0.7531, 0.7634, 0.7767, 0.7946, 0.8203, 0.8618},
		{0.8491, 0.8609, 0.8762, 0.8968, 0.9265, 0.9745},
		{0.945, 0.9584, 0.9757, 0.999, 1.0327, 1.0872},
		{1.041, 1.0559, 1.0752, 1.1013, 1.1389, 1.1999},
		{1.1369, 1.1534, 1.1747, 1.2035, 1.2451, 1.3126},
		{1.1496, 1.1676, 1.1909, 1.2224, 1.2679, 1.342},
		{1.1403, 1.1575, 1.18, 1.2105, 1.2549, 1.3279},
		{1.1266, 1.1422, 1.1625, 1.19, 1.2303, 1.2969},
		{1.1122, 1.1259, 1.1435, 1.1675, 1.2024, 1.26},
		{1.0977, 1.1093, 1.1243, 1.1444, 1.1736, 1.2211},
		{1.0832, 1.0928, 1.105, 1.1213, 1.1446, 1.1817},
		{1.0687, 1.0762, 1.0857, 1.0981, 1.1155, 1.142},
		{1.0541, 1.0596, 1.0664, 1.075, 1.0864, 1.1024},
		{1.0396, 1.043, 1.0471, 1.0518, 1.0572, 1.0627},
		{1.0251, 1.0265, 1.0278, 1.0286, 1.0281, 1.023},
		{1.0105, 1.0099, 1.0085, 1.0055, 0.999, 0.9833},
		{0.996, 0.9933, 0.9892, 0.9823, 0.9699, 0.9436},
		{0.9815, 0.9767, 0.9698, 0.9591, 0.9408, 0.9038},
		{0.9779, 0.9723, 0.9641, 0.9514, 0.9296, 0.8859},
		{0.9786, 0.9731, 0.9649, 0.9521, 0.93, 0.8848},
		{0.9805, 0.9754, 0.9679, 0.956, 0.9354, 0.8926},
		{0.9827, 0.9781, 0.9715, 0.9609, 0.9425, 0.9043},
		{0.9849, 0.9809, 0.9752, 0.9661, 0.9503, 0.9175},
		{0.9871, 0.9838, 0.9789, 0.9713, 0.9582, 0.9312},
		{0.9893, 0.9866, 0.9827, 0.9766, 0.9662, 0.9451},
		{0.9915, 0.9894, 0.9864, 0.9818, 0.9741, 0.959},
		{0.9937, 0.9922, 0.9901, 0.9871, 0.9821, 0.973},
		{0.9959, 0.995, 0.9939, 0.9923, 0.9901, 0.987},
		{0.9981, 0.9978, 0.9976, 0.9976, 0.9981, 1.001},
	},
	ac  = {
		{0.9337, 0.9352, 0.937, 0.9393, 0.9423, 0.9466},
		{0.841, 0.8419, 0.843, 0.8445, 0.8467, 0.8504},
		{0.7437, 0.743, 0.7422, 0.7413, 0.7401, 0.7388},
		{0.6456, 0.6432, 0.6401, 0.636, 0.6304, 0.622},
		{0.5474, 0.5431, 0.5376, 0.5302, 0.5198, 0.5036},
		{0.4492, 0.443, 0.4351, 0.4244, 0.4091, 0.3846},
		{0.351, 0.3429, 0.3326, 0.3186, 0.2983, 0.2654},
		{0.2527, 0.2429, 0.2301, 0.2127, 0.1875, 0.1464},
		{0.1545, 0.1428, 0.1276, 0.107, 0.0769, 0.0276},
		{0.0565, 0.043, 0.0255, 0.0017, -0.033, -0.0898},
		{-0.0403, -0.0554, -0.0749, -0.1013, -0.1398, -0.2027},
		{-0.1281, -0.1439, -0.1644, -0.1923, -0.2329, -0.2994},
		{-0.1484, -0.1662, -0.1892, -0.2203, -0.2653, -0.3385},
		{-0.1424, -0.16, -0.183, -0.2143, -0.2599, -0.3348},
		{-0.1294, -0.1456, -0.1667, -0.1956, -0.2381, -0.3088},
		{-0.1149, -0.1291, -0.1477, -0.1731, -0.2105, -0.2729},
		{-0.1, -0.1122, -0.128, -0.1495, -0.1809, -0.2332},
		{-0.0852, -0.0952, -0.1082, -0.1256, -0.1508, -0.1919},
		{-0.0703, -0.0782, -0.0883, -0.1016, -0.1205, -0.1502},
		{-0.0554, -0.0612, -0.0684, -0.0776, -0.0901, -0.1083},
		{-0.0405, -0.0442, -0.0485, -0.0537, -0.0597, -0.0663},
		{-0.0257, -0.0272, -0.0286, -0.0297, -0.0294, -0.0245},
		{-0.0108, -0.0102, -0.0088, -0.0058, 0.0008, 0.017},
		{0.0039, 0.0066, 0.0107, 0.0177, 0.0302, 0.0573},
		{0.0174, 0.0219, 0.0285, 0.0388, 0.0566, 0.0929},
	},
}

@(private = "file")
HS_AIII := HS_Table {
	phi = {0.0322, 0.1293, 0.2023, 0.2605, 0.3214, 0.3785},
	sdr = {0.2952, 0.2317, 0.2106, 0.2028, 0.1977, 0.1907},
	sd0 = {0.3727, 0.618, 0.7935, 0.9373, 1.062, 1.1736},
	sde = {0.3744, 0.6317, 0.8087, 0.9365, 1.0331, 1.0992},
	irf = {
		{0.3333, 0.1667, 0.1111, 0.0833, 0.0667, 0.0556},
		{0.6774, 0.3549, 0.2447, 0.1884, 0.1548, 0.1321},
		{1.0218, 0.5459, 0.3828, 0.2991, 0.2497, 0.2167},
		{1.0329, 0.7373, 0.5219, 0.4113, 0.3469, 0.3042},
		{1.0333, 0.9287, 0.6611, 0.5238, 0.4448, 0.3929},
		{1.0333, 1.1201, 0.8004, 0.6365, 0.543, 0.4821},
		{1.0333, 1.1449, 0.9397, 0.7492, 0.6412, 0.5713},
		{1.0333, 1.1481, 1.079, 0.8618, 0.7394, 0.6607},
		{1.0333, 1.1485, 1.2183, 0.9745, 0.8376, 0.7501},
		{1.0333, 1.1486, 1.2465, 1.0872, 0.9359, 0.8395},
		{1.0333, 1.1486, 1.2522, 1.1999, 1.0341, 0.9288},
		{1.0333, 1.1486, 1.2533, 1.3126, 1.1323, 1.0182},
		{1.0333, 1.1486, 1.2536, 1.342, 1.2306, 1.1076},
		{1.0226, 1.127, 1.2311, 1.3279, 1.3074, 1.176},
		{1.0111, 1.0999, 1.1996, 1.2969, 1.3704, 1.2284},
		{0.9997, 1.0717, 1.1652, 1.26, 1.3602, 1.2718},
		{0.9989, 1.0433, 1.1302, 1.2211, 1.3256, 1.3107},
		{0.9989, 1.0148, 1.0949, 1.1817, 1.2831, 1.3474},
		{0.9989, 0.9864, 1.0596, 1.142, 1.2378, 1.3275},
		{0.9989, 0.9795, 1.0242, 1.1024, 1.1918, 1.2862},
		{0.9989, 0.9782, 0.9889, 1.0627, 1.1454, 1.2368},
		{0.9989, 0.978, 0.9536, 1.023, 1.0989, 1.1842},
		{0.9989, 0.9779, 0.9407, 0.9833, 1.0524, 1.1305},
		{0.9989, 0.9779, 0.937, 0.9436, 1.0059, 1.0763},
		{0.9989, 0.9779, 0.936, 0.9038, 0.9594, 1.022},
		{0.9989, 0.9779, 0.9358, 0.8859, 0.9128, 0.9676},
		{0.9992, 0.9807, 0.9402, 0.8848, 0.8732, 0.9211},
		{0.9996, 0.9846, 0.9475, 0.8926, 0.8402, 0.8837},
		{1.0, 0.9887, 0.956, 0.9043, 0.8329, 0.8531},
		{1.0, 0.9929, 0.9648, 0.9175, 0.8417, 0.8268},
		{1.0, 0.9972, 0.9737, 0.9312, 0.8581, 0.803},
		{1.0, 1.0014, 0.9826, 0.9451, 0.878, 0.8015},
		{1.0, 1.0028, 0.9916, 0.959, 0.8992, 0.8165},
		{1.0, 1.0032, 1.0005, 0.973, 0.9209, 0.8409},
		{1.0, 1.0033, 1.0095, 0.987, 0.9428, 0.8701},
		{1.0, 1.0033, 1.0139, 1.001, 0.9648, 0.9014},
	},
	ac  = {
		{0.6806, 0.8661, 0.9239, 0.9466, 0.9585, 0.9666},
		{0.341, 0.697, 0.8155, 0.8504, 0.8794, 0.901},
		{0.011, 0.5233, 0.6995, 0.7388, 0.7761, 0.8178},
		{0.0004, 0.3493, 0.5787, 0.622, 0.664, 0.7216},
		{0.0, 0.1775, 0.4416, 0.5036, 0.5486, 0.6116},
		{0.0, 0.023, 0.2981, 0.3846, 0.4321, 0.4861},
		{0.0, 0.0, 0.1532, 0.2654, 0.3152, 0.3547},
		{-0.0, -0.023, 0.0112, 0.1464, 0.1983, 0.2234},
		{-0.0, -0.0481, -0.1148, 0.0276, 0.0818, 0.0979},
		{-0.0004, -0.0739, -0.1647, -0.0898, -0.0335, -0.0202},
		{-0.011, -0.0997, -0.1983, -0.2027, -0.1455, -0.1304},
		{-0.0223, -0.1249, -0.227, -0.2994, -0.2483, -0.2311},
		{-0.033, -0.1455, -0.2482, -0.3385, -0.3352, -0.3159},
		{-0.023, -0.1309, -0.2371, -0.3348, -0.3873, -0.3692},
		{-0.0117, -0.1071, -0.213, -0.3088, -0.4071, -0.398},
		{-0.0007, -0.0815, -0.1846, -0.2729, -0.3802, -0.4097},
		{-0.0, -0.0557, -0.1544, -0.2332, -0.3356, -0.407},
		{-0.0, -0.0302, -0.1206, -0.1919, -0.2842, -0.3855},
		{-0.0, -0.0069, -0.0847, -0.1502, -0.2302, -0.3299},
		{-0.0, -0.0009, -0.0481, -0.1083, -0.1753, -0.2591},
		{0.0, 0.0029, -0.012, -0.0663, -0.1201, -0.1826},
		{0.0, 0.0066, 0.0208, -0.0245, -0.0649, -0.1062},
		{0.0, 0.0104, 0.0375, 0.017, -0.0101, -0.0326},
		{0.0004, 0.0142, 0.0477, 0.0573, 0.0435, 0.037},
		{0.0007, 0.018, 0.0556, 0.0929, 0.0938, 0.1015},
	},
}

hong_stein_reference_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Hong & Stein (1999) Reference Checks ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	base := bfin.HONG_STEIN_1999

	// 1. newswatchers only: the news enters linearly over z periods, dP an MA(z - 1)
	// of the news with autocovariances (z - k) sigma^2 / z^2, pricing error sd
	// sigma sqrt(sum_{i<z} (i / z)^2) (0.9373 for z = 12, sigma = 0.5)
	{
		h := bfin.hong_stein_impulse(0, base, 20)
		a := bfin.hong_stein_autocovariances(0, base, 14)
		defer {delete(h); delete(a)}
		ok := true
		for k in 0 ..< 20 {ok &&= h[k] == (1.0 / 12 if k < 12 else 0)}
		for k in 0 ..= 14 {ok &&= abs(a[k] - 0.25 * f64(max(12 - k, 0)) / 144) < 1e-16}
		ss := 0.0
		for i in 1 ..< 12 {ss += f64(i * i) / 144}
		e0 := bfin.hong_stein_pricing_error_sd(0, base)
		expect(
			&passed,
			&total,
			ok && abs(e0 - 0.5 * math.sqrt(ss)) < 1e-15,
			fmt.tprintf(
				"newswatchers only: news enters 1/z a period for z periods, MA autocovariances (z-k) s^2/z^2, pricing error sd %.4f (paper 0.9373)",
				e0,
			),
		)
	}

	// 2. stationarity: for j = 1 the AR roots have modulus sqrt(phi) (0 < phi < 4);
	// Lemma A.1: |phi| < 1 is necessary for every j
	{
		ok := abs(bfin.hong_stein_ar_radius(0.64, 1) - 0.8) < 1e-12
		ok &&= abs(bfin.hong_stein_ar_radius(0.25, 1) - 0.5) < 1e-12
		for j in 1 ..= 18 {ok &&= !bfin.hong_stein_stationary(1.0, j) && !bfin.hong_stein_stationary(-1.0, j)}
		ok &&= bfin.hong_stein_stationary(0.2605, 12) && bfin.hong_stein_stationary(0.99, 1)
		expect(
			&passed,
			&total,
			ok,
			"AR radius sqrt(phi) for j = 1, |phi| = 1 never stationary (Lemma A.1, j = 1 .. 18), the base equilibrium stationary",
		)
	}

	// 3. autocovariances off equilibrium = statsmodels arma_acovf(ar = [1, -phi, 0, ..,
	// phi], ma = ones(z) / z, sigma2) (scratch: a separate implementation)
	{
		SM1 := [11]f64 {
			0.19626849315068493,
			0.1637808219178082,
			0.10793424657534247,
			0.04508,
			-0.01601024657534245,
			-0.0636836219178082,
			-0.06823933315068492,
			-0.05285207391780821,
			-0.029379622175342464,
			-0.004010812680000004,
			0.017901842771342456,
		}
		SM2 := [11]f64 {
			0.5582137161084529,
			0.36097820308346623,
			0.02235513024986708,
			-0.15238038277511962,
			-0.07863098086124401,
			0.03318723086124402,
			0.05031819527511962,
			0.007708933986244022,
			-0.01917416757999402,
			-0.01209739570480712,
			0.0031845473438341055,
		}
		a1 := bfin.hong_stein_autocovariances(0.3, {5, 4, 1, 0.7}, 10)
		a2 := bfin.hong_stein_autocovariances(0.45, {3, 1, 1, 1.0}, 10)
		defer {delete(a1); delete(a2)}
		err := 0.0
		for k in 0 ..= 10 {err = max(err, abs(a1[k] - SM1[k]), abs(a2[k] - SM2[k]))}
		expect(
			&passed,
			&total,
			err < 1e-14,
			fmt.tprintf(
				"autocovariances (phi 0.3, z 5, j 4; phi 0.45, z 3, j 1) = statsmodels arma_acovf, max diff %.1e",
				err,
			),
		)
	}

	// 4. Tables AI - AIII. Printed to 4 decimals: phi, impulse responses and
	// autocorrelations within 6e-5, the standard deviations within 1.5e-4 (the paper's
	// sd of returns for j = 9 is 0.2179 against 0.2180 here). One entry is left out:
	// the pricing-error sd for gamma = 1/13 is printed as 0.9070, here 0.9007 (the
	// neighbours 0.8998, 0.8999 agree to the digit; most likely transposed digits)
	{
		Col :: struct {
			z, j:  int,
			gamma: f64,
		}
		sets := [3][6]Col {
			{
				{12, 3, 1.0 / 3},
				{12, 6, 1.0 / 3},
				{12, 9, 1.0 / 3},
				{12, 12, 1.0 / 3},
				{12, 15, 1.0 / 3},
				{12, 18, 1.0 / 3},
			},
			{
				{12, 12, 1.0 / 13},
				{12, 12, 1.0 / 11},
				{12, 12, 1.0 / 9},
				{12, 12, 1.0 / 7},
				{12, 12, 1.0 / 5},
				{12, 12, 1.0 / 3},
			},
			{
				{3, 12, 1.0 / 3},
				{6, 12, 1.0 / 3},
				{9, 12, 1.0 / 3},
				{12, 12, 1.0 / 3},
				{15, 12, 1.0 / 3},
				{18, 12, 1.0 / 3},
			},
		}
		tabs := [3]^HS_Table{&HS_AI, &HS_AII, &HS_AIII}
		names := [3]string{"AI (j 3 - 18)", "AII (gamma 1/13 - 1/3)", "AIII (z 3 - 18)"}
		props := true
		odd := 0.0
		for T, ti in tabs {
			dphi, dsd, dirf, dac := 0.0, 0.0, 0.0, 0.0
			ok := true
			for c, ci in sets[ti] {
				p := bfin.Hong_Stein_Params{c.z, c.j, c.gamma, 0.5}
				phi, found := bfin.hong_stein_equilibrium(p)
				ok &&= found
				a := bfin.hong_stein_autocovariances(phi, p, 25)
				h := bfin.hong_stein_impulse(phi, p, 400)
				defer {delete(a); delete(h)}
				sde := bfin.hong_stein_pricing_error_sd(phi, p)
				sd0 := bfin.hong_stein_pricing_error_sd(0, p)
				dphi = max(dphi, abs(phi - T.phi[ci]))
				dsd = max(dsd, abs(math.sqrt(a[0]) - T.sdr[ci]), abs(sd0 - T.sd0[ci]))
				if ti == 1 && ci == 0 {odd = sde} else {dsd = max(dsd, abs(sde - T.sde[ci]))}
				C, peak, at := 0.0, 0.0, 0
				for k in 0 ..< 400 {
					C += h[k]
					if k < 36 {dirf = max(dirf, abs(C - T.irf[k][ci]))}
					if C > peak {peak, at = C, k}
				}
				for k in 1 ..= 25 {dac = max(dac, abs(a[k] / a[0] - T.ac[k - 1][ci]))}
				// Lemma 1: phi > 0; Proposition 1: the peak exceeds 1, at lag j when
				// j >= z - 1 and no earlier than j otherwise
				props &&= phi > 0 && peak > 1 && (at == c.j if c.j >= c.z - 1 else at >= c.j)
			}
			expect(
				&passed,
				&total,
				ok && dphi < 6e-5 && dsd < 1.5e-4 && dirf < 6e-5 && dac < 6e-5,
				fmt.tprintf(
					"Table %s: phi within %.1e, sds within %.1e, impulse responses (lags 0-35) within %.1e, autocorrelations (1-25) within %.1e",
					names[ti],
					dphi,
					dsd,
					dirf,
					dac,
				),
			)
		}
		expect(
			&passed,
			&total,
			props,
			fmt.tprintf(
				"Lemma 1 and Proposition 1 on all 18: phi > 0, the impulse response peaks above 1, at lag j when j >= z - 1, later otherwise (gamma 1/13 pricing error sd %.4f, printed 0.9070, left out)",
				odd,
			),
		)
	}

	// 5. the equilibrium is a fixed point of the best response; Proposition 2:
	// risk-neutral momentum traders (cov(P_{t+j} - P_t, dP_{t-1}) = 0) make some
	// autocovariance at lags 2 .. j + 1 negative
	{
		phi, _ := bfin.hong_stein_equilibrium(base)
		fixed := abs(bfin.hong_stein_best_response(phi, base) - phi)
		ok := fixed < 1e-10
		zs := [4][2]int{{12, 12}, {6, 6}, {12, 6}, {3, 1}}
		phis: [4]f64
		for zj, i in zs {
			p := bfin.Hong_Stein_Params{zj[0], zj[1], math.inf_f64(1), 0.5}
			f, found := bfin.hong_stein_equilibrium(p)
			phis[i] = f
			a := bfin.hong_stein_autocovariances(f, p, zj[1] + 1)
			defer delete(a)
			cov, neg := 0.0, false
			for k in 2 ..= zj[1] + 1 {cov += a[k]; neg ||= a[k] < 0}
			ok &&= found && abs(cov) < 1e-12 * a[0] && neg
		}
		expect(
			&passed,
			&total,
			ok && abs(phis[3] - 0.5) < 1e-9,
			fmt.tprintf(
				"best response(phi*) = phi* within %.1e; Proposition 2, risk-neutral momentum traders (z, j) = (12, 12), (6, 6), (12, 6), (3, 1): phi %.4f %.4f %.4f %.4f, cov = 0 and a negative autocovariance within j + 1 lags",
				fixed,
				phis[0],
				phis[1],
				phis[2],
				phis[3],
			),
		)
	}

	fmt.printf("Hong-Stein reference: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}

hong_stein_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== Hong & Stein (1999): Momentum and Reversal from One Market ===")
	context.allocator = allocator
	passed, total := 0, 0
	expect :: proc(passed, total: ^int, ok: bool, name: string) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}

	// three markets of N independent stocks, monthly news ~ N(0, sigma): Hong-Stein
	// (z = j = 12, gamma = 1/3, phi at equilibrium), newswatchers only (phi = 0) and
	// rational (z = 1: prices are a random walk). Price changes are the returns.
	N, BURN, M :: 2000, 120, 1200
	base := bfin.HONG_STEIN_1999
	phi, _ := bfin.hong_stein_equilibrium(base)
	markets := [3]struct {
		p:   bfin.Hong_Stein_Params,
		phi: f64,
	}{{base, phi}, {base, 0}, {{1, base.j, base.gamma, base.sigma}, 0}}
	LAGS :: 48 // Newey-West lags: monthly returns here are autocorrelated for ~40 months
	res: [3][4]f64 // [market][JT mean, t, DBT losers - winners mean, t]
	acf_ok := true
	acf_worst := 0.0
	acf_lags := [5]int{1, 6, 12, 18, 24}
	acf_sim, acf_th: [5]f64
	for mk, k in markets {
		R := make([][]f64, N)
		e := make([]f64, BURN + M)
		defer {free_panel(R); delete(e)}
		for i in 0 ..< N {
			for &v in e {v = rand.float64_normal(0, mk.p.sigma)}
			dP := bfin.hong_stein_simulate(e, mk.phi, mk.p)
			R[i] = make([]f64, M)
			copy(R[i], dP[BURN:])
			delete(dP)
		}
		if k == 0 {
			// sample autocorrelations against the model's: the pooled ratio
			// sum_i A_i / sum_i B_i (A_i the lag-L autocovariance of stock i, B_i its
			// variance, both about the known mean 0), within 4 delta-method standard
			// errors (across stocks, of (A_i - r B_i) / mean B). Averaging the stocks'
			// own ratios instead is biased by O(1 / months), ~0.003 here, which 1000
			// stocks resolve.
			a := bfin.hong_stein_autocovariances(mk.phi, mk.p, 24)
			defer delete(a)
			A := make([]f64, N)
			B := make([]f64, N)
			defer {delete(A); delete(B)}
			for L, li in acf_lags {
				sa, sb := 0.0, 0.0
				for i in 0 ..< N {
					B[i], A[i] = 0, 0
					for t in 0 ..< M {B[i] += R[i][t] * R[i][t] / M}
					for t in 0 ..< M - L {A[i] += R[i][t] * R[i][t + L] / f64(M - L)}
					sa += A[i]
					sb += B[i]
				}
				m := sa / sb
				sd := 0.0
				for i in 0 ..< N {
					d := (A[i] - m * B[i]) / (sb / N)
					sd += d * d / (N - 1)
				}
				se := math.sqrt(sd / N)
				acf_sim[li], acf_th[li] = m, a[L] / a[0]
				acf_ok &&= abs(m - acf_th[li]) < 4 * se
				acf_worst = max(acf_worst, abs(m - acf_th[li]) / se)
			}
		}
		Rt := transpose_panel(R)
		defer free_panel(Rt)
		// JT 6 / 6: deciles of the past six months' price change, held six months
		PR := bfin.past_return(R, 6, kind = .Sum)
		W := long_short_weights(PR)
		res[k][0], res[k][1] = overlapping_spread_t(W, Rt, 6, 12, LAGS)
		free_panel(W)
		bfin.signal_panel_free(PR)
		// DBT 36 / 36: deciles of the past 36 months, market-adjusted, held 36 months
		mkt := make([]f64, M)
		defer delete(mkt)
		for t in 0 ..< M {for i in 0 ..< N {mkt[t] += R[i][t] / N}}
		CU := bfin.past_return(R, 36, market = mkt, kind = .Sum)
		W = long_short_weights(CU)
		m, t := overlapping_spread_t(W, Rt, 36, 72, LAGS)
		res[k][2], res[k][3] = -m, -t
		free_panel(W)
		bfin.signal_panel_free(CU)
	}
	expect(
		&passed,
		&total,
		acf_ok,
		fmt.tprintf(
			"Hong-Stein market (phi %.4f), %d stocks x %d months: return autocorrelations at lags 1, 6, 12, 18, 24 %.3f %.3f %.3f %.3f %.3f vs the model's %.3f %.3f %.3f %.3f %.3f (within %.1f se)",
			phi,
			N,
			M,
			acf_sim[0],
			acf_sim[1],
			acf_sim[2],
			acf_sim[3],
			acf_sim[4],
			acf_th[0],
			acf_th[1],
			acf_th[2],
			acf_th[3],
			acf_th[4],
			acf_worst,
		),
	)
	expect(
		&passed,
		&total,
		res[0][1] > 2 && res[0][3] > 2,
		fmt.tprintf(
			"Hong-Stein market: JT 6/6 winners - losers %.3f a month (t %.1f) and DBT 36/36 losers - winners %.3f (t %.1f): momentum and long-term reversal together",
			res[0][0],
			res[0][1],
			res[0][2],
			res[0][3],
		),
	)
	expect(
		&passed,
		&total,
		res[1][1] > 2 && res[1][3] < -2,
		fmt.tprintf(
			"newswatchers only (phi = 0): JT %.3f (t %.1f), DBT losers - winners %.3f (t %.1f): momentum but no reversal (prices never overshoot)",
			res[1][0],
			res[1][1],
			res[1][2],
			res[1][3],
		),
	)
	expect(
		&passed,
		&total,
		abs(res[2][1]) < 3 && abs(res[2][3]) < 3,
		fmt.tprintf(
			"rational market (random walk): JT %.3f (t %.1f), DBT %.3f (t %.1f)",
			res[2][0],
			res[2][1],
			res[2][2],
			res[2][3],
		),
	)

	fmt.printf("Hong-Stein: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
