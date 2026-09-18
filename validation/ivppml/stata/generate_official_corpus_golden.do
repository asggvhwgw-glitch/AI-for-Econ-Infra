version 16.0
clear all
set more off
args root datadir
if `"`root'"' == "" local root ".."
if `"`datadir'"' == "" local datadir `"`root'/official"'

tempname H HV
postfile `H' str16 model str16 term double b se N df_a num_sep num_sep_mu iters using `"`root'/official_stata_golden.dta"', replace
postfile `HV' str16 model str16 row str16 col double v using `"`root'/official_stata_golden_vcov.dta"', replace

capture program drop _dump_official
program define _dump_official
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

use `"`datadir'/ivppmlhdfe_ClassA.dta"', clear
quietly ivppmlhdfe y x2 (x1 = z), absorb(id year) vce(robust)
_dump_official `H' `HV' ClassA

use `"`datadir'/ivppmlhdfe_ClassB.dta"', clear
quietly ivppmlhdfe y x2 (x1 = z), absorb(exp#year imp#year) vce(cluster pair)
_dump_official `H' `HV' ClassB

use `"`datadir'/ivppmlhdfe_ClassC.dta"', clear
quietly ivppmlhdfe y x2 (x1 = z), absorb(exp#imp exp#year imp#year) vce(cluster pair) separation(all) standardize
_dump_official `H' `HV' ClassC

postclose `H'
postclose `HV'
use `"`root'/official_stata_golden.dta"', clear
export delimited using `"`root'/official_stata_golden.csv"', replace
use `"`root'/official_stata_golden_vcov.dta"', clear
export delimited using `"`root'/official_stata_golden_vcov.csv"', replace
