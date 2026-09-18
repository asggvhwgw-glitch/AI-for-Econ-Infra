version 16.0
clear all
set more off

args root
if `"`root'"' == "" local root ".."
use `"`root'/fixtures/ivppml_feature_fixture.dta"', clear

tempname H HV
postfile `H' str24 model str24 term double b se N df_a num_sep num_sep_mu iters using `"`root'/stata_golden.dta"', replace
postfile `HV' str24 model str24 row str24 col double v using `"`root'/stata_golden_vcov.dta"', replace

capture program drop _dump_ivppml
program define _dump_ivppml
    args H HV model
    tempname V dfa nsep nmu it
    matrix `V' = e(V)
    local cn : colnames `V'
    scalar `dfa' = 0
    capture scalar `dfa' = e(df_a)
    scalar `nsep' = 0
    capture scalar `nsep' = e(num_separated)
    scalar `nmu' = 0
    capture scalar `nmu' = e(num_sep_mu)
    scalar `it' = .
    capture scalar `it' = e(ic)
    foreach t of local cn {
        post `H' ("`model'") ("`t'") (_b[`t']) (_se[`t']) (e(N)) (`dfa') (`nsep') (`nmu') (`it')
    }
    local nr = rowsof(`V')
    forvalues i=1/`nr' {
        local ri : word `i' of `cn'
        forvalues j=1/`nr' {
            local cj : word `j' of `cn'
            post `HV' ("`model'") ("`ri'") ("`cj'") (`V'[`i',`j'])
        }
    }
end

quietly ivppmlhdfe y c (e = z), absorb(g1 g2) vce(robust)
_dump_ivppml `H' `HV' m01_base

quietly ivppmlhdfe y c (e = z), absorb(g1 g2) vce(cluster cl1 cl2)
_dump_ivppml `H' `HV' m02_cluster2

quietly ivppmlhdfe y c (e = z), absorb(g1 g2) vce(robust) standardize
_dump_ivppml `H' `HV' m03_standardize

quietly ivppmlhdfe y c (e = z) [fw=fw], absorb(g1 g2) vce(robust)
_dump_ivppml `H' `HV' m04_fweight

quietly ivppmlhdfe y c (e = z) [pw=pw], absorb(g1 g2) vce(robust)
_dump_ivppml `H' `HV' m05_pweight

quietly ivppmlhdfe yexp c (e = z), absorb(g1 g2) exposure(expo) vce(robust)
_dump_ivppml `H' `HV' m06_exposure

quietly ivppmlhdfe y c (e e2 = z z2 z3), absorb(g1 g2) vce(robust)
_dump_ivppml `H' `HV' m07_multiendog

quietly ivppmlhdfe y c (e = z), absorb(g1 g2) vce(robust) separation(all) tagsep(_ivsep)
_dump_ivppml `H' `HV' m08_mu
capture gen long _row = _n
capture gen byte _esample = e(sample)
export delimited _row _esample _ivsep using `"`root'/stata_mu_mask.csv"', replace

postclose `H'
postclose `HV'
use `"`root'/stata_golden.dta"', clear
export delimited using `"`root'/stata_golden.csv"', replace
use `"`root'/stata_golden_vcov.dta"', clear
export delimited using `"`root'/stata_golden_vcov.csv"', replace
