version 16.0
clear all
set more off

args root
if `"`root'"' == "" local root ".."
local fixture `"`root'/fixtures/ppml_fixture.dta"'
local outfile `"`root'/stata_golden.csv"'

use `"`fixture'"', clear

tempname H
postfile `H' str24 model str16 term double b se N df_a num_sep iters using `"`root'/stata_golden.dta"', replace

capture program drop _post_ppml
program define _post_ppml
    args H model term
    tempname dfa nsep it
    scalar `dfa' = 0
    capture scalar `dfa' = e(df_a)
    scalar `nsep' = 0
    capture scalar `nsep' = e(num_separated)
    scalar `it' = .
    capture scalar `it' = e(ic)
    post `H' ("`model'") ("`term'") (_b[`term']) (_se[`term']) (e(N)) (`dfa') (`nsep') (`it')
end

quietly ppmlhdfe y x1 x2, noabsorb vce(robust) separation(fe simplex relu)
_post_ppml `H' m01_no_fe x1
_post_ppml `H' m01_no_fe x2
_post_ppml `H' m01_no_fe _cons

quietly ppmlhdfe y x1 x2, absorb(g1) vce(robust) separation(fe simplex relu)
_post_ppml `H' m02_one_fe x1
_post_ppml `H' m02_one_fe x2

quietly ppmlhdfe y x1 x2, absorb(g1 g2) vce(cluster cl1) separation(fe simplex relu)
_post_ppml `H' m03_two_fe_cluster x1
_post_ppml `H' m03_two_fe_cluster x2

quietly ppmlhdfe y x1 x2, absorb(g1 g2 g3) vce(cluster cl1 cl2) separation(fe simplex relu)
_post_ppml `H' m04_three_fe_2cluster x1
_post_ppml `H' m04_three_fe_2cluster x2

quietly ppmlhdfe yexp x1 x2, absorb(g1 g2) exposure(expo) vce(robust) separation(fe simplex relu)
_post_ppml `H' m05_exposure x1
_post_ppml `H' m05_exposure x2

postclose `H'
use `"`root'/stata_golden.dta"', clear
export delimited using `"`outfile'"', replace

* Official primer: tag the observation-level separation mask.
use `"`root'/fixtures/separation_primer.dta"', clear
quietly ppmlhdfe y, absorb(id1 id2) tagsep(separated) zvar(z)
export delimited y id1 id2 separated z using `"`root'/stata_separation_primer.csv"', replace
