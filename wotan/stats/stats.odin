// Classical statistics. This first part: probability distributions and the
// special functions behind them.
//
// Continuous: Normal, Log_Normal, Student_T, Chi_Squared, Fisher_F, Gamma,
// Beta, Exponential. Discrete: Binomial, Poisson, Negative_Binomial.
// Each distribution is a struct of its parameters, for example
// stats.Student_T{df = 5}, with procedures named after it
// (student_t_cdf(d, x), ...) and the groups below, which take any of them:
//
//     d := stats.Student_T{df = 5}
//     p := stats.sf(d, 2.3)          // P(T > 2.3)
//     c := stats.ppf(d, 0.975)       // the 97.5% quantile
//
// Values match scipy.stats to about 1e-13 relative error, in the far tails
// as well (sf and isf work on the upper tail directly). See the stats guide.
package stats

pdf :: proc {
	normal_pdf,
	log_normal_pdf,
	student_t_pdf,
	chi_squared_pdf,
	fisher_f_pdf,
	gamma_pdf,
	beta_pdf,
	exponential_pdf,
}

logpdf :: proc {
	normal_logpdf,
	log_normal_logpdf,
	student_t_logpdf,
	chi_squared_logpdf,
	fisher_f_logpdf,
	gamma_logpdf,
	beta_logpdf,
	exponential_logpdf,
}

pmf :: proc {
	binomial_pmf,
	poisson_pmf,
	negative_binomial_pmf,
}

logpmf :: proc {
	binomial_logpmf,
	poisson_logpmf,
	negative_binomial_logpmf,
}

cdf :: proc {
	normal_cdf,
	log_normal_cdf,
	student_t_cdf,
	chi_squared_cdf,
	fisher_f_cdf,
	studentized_range_cdf,
	gamma_cdf,
	beta_cdf,
	exponential_cdf,
	binomial_cdf,
	poisson_cdf,
	negative_binomial_cdf,
}

sf :: proc {
	normal_sf,
	log_normal_sf,
	student_t_sf,
	chi_squared_sf,
	fisher_f_sf,
	studentized_range_sf,
	gamma_sf,
	beta_sf,
	exponential_sf,
	binomial_sf,
	poisson_sf,
	negative_binomial_sf,
}

ppf :: proc {
	normal_ppf,
	log_normal_ppf,
	student_t_ppf,
	chi_squared_ppf,
	fisher_f_ppf,
	studentized_range_ppf,
	gamma_ppf,
	beta_ppf,
	exponential_ppf,
	binomial_ppf,
	poisson_ppf,
	negative_binomial_ppf,
}

isf :: proc {
	normal_isf,
	log_normal_isf,
	student_t_isf,
	chi_squared_isf,
	fisher_f_isf,
	studentized_range_isf,
	gamma_isf,
	beta_isf,
	exponential_isf,
	binomial_isf,
	poisson_isf,
	negative_binomial_isf,
}

mean :: proc {
	normal_mean,
	log_normal_mean,
	student_t_mean,
	chi_squared_mean,
	fisher_f_mean,
	gamma_mean,
	beta_mean,
	exponential_mean,
	binomial_mean,
	poisson_mean,
	negative_binomial_mean,
}

variance :: proc {
	normal_variance,
	log_normal_variance,
	student_t_variance,
	chi_squared_variance,
	fisher_f_variance,
	gamma_variance,
	beta_variance,
	exponential_variance,
	binomial_variance,
	poisson_variance,
	negative_binomial_variance,
}

random :: proc {
	normal_random,
	log_normal_random,
	student_t_random,
	chi_squared_random,
	fisher_f_random,
	gamma_random,
	beta_random,
	exponential_random,
	binomial_random,
	poisson_random,
	negative_binomial_random,
}

sample :: proc {
	normal_sample,
	log_normal_sample,
	student_t_sample,
	chi_squared_sample,
	fisher_f_sample,
	gamma_sample,
	beta_sample,
	exponential_sample,
	binomial_sample,
	poisson_sample,
	negative_binomial_sample,
}
