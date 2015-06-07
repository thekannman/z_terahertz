//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

// A set of functions for calculation of electric field / frequency maps
// as outlined in the papers of Skinner et al. See, for example, S.M. Gruenbaum,
// et al. J. Chem. Theory Comput. 2013 9 (7), 3109. These maps are necessary
// for calculation of vibrational spectra using the methodology  outline in the
// aforementioned papers.

// Abbreviations:
//   FTIR = fourier-transform infrared
//   SFG = sum-frequency generation

#include <armadillo>

#ifndef _Z_MAP_HPP_
#define _Z_MAP_HPP_

// Calculates switching function needed for SFG calculations in slab geometry.
// Allows smooth switching between contribution to upper and lower surface as
// one moves through the slab.
extern double SwitchFunction(const double z, const double u_z,
                             const double dipPos, const double zcenter,
                             const double zboxL, const double rswitch,
                             const double rswitch2, const double rswitch3);

// Electric field / frequency map for three-site water models (e.g., SPC/E)
extern void ThreeSiteMap(const double eh, const arma::rowvec& u,
                         double& omega01, double& chi01, arma::rowvec& mu01,
                         double& muprime, arma::rowvec& alphaDiag,
                         arma::rowvec& alphaOffDiag, const bool doOH);

// Electric field / frequency map for four-site water models (e.g., TIP4P
// and E3B) as needed for Raman or FTIR spectra calculations.
extern void FourSiteMap(const double eh, const arma::rowvec& u, double& omega01,
                        double& chi01, arma::rowvec& mu01, double& muprime,
                        arma::rowvec& alphaDiag, arma::rowvec& alphaOffDiag,
                        const bool doOH);

// Electric field / frequency map for four-site water models (e.g., TIP4P
// and E3B) as needed for FTIR (not Raman) spectra calculations.
extern void FourSiteMapIR(const double eh, const arma::rowvec& u,
                           double& omega01, double& chi01, arma::rowvec& mu01,
                           double& muprime, const bool doOH);

// Electric field / frequency map for four-site water models (e.g., TIP4P
// and E3B) as needed for 2DIR spectra calculations.
extern void FourSiteMap2DIR(const double eh, double& omega01, double& omega12,
                             double& chi01, double& chi12, double& mu01,
                             double& mu12, double& muprime,
                             const bool doOH);

// Electric field / frequency map for four-site water models (e.g., TIP4P
// and E3B) as needed for SFG spectra calculations.
extern void FourSiteMapSFG(const double eh, arma::rowvec u,
                            const double switcher, double& omega01,
                            double& mu01, double& alpha, const double avgFreq);

// Calculates the intramolecular coupling between OH groups for inclusion
// in the system Hamiltonian needed for coupled spectra calculations
extern double IntraCouple(const double eh1, const double eh2,
                          const double omega011, const double omega012,
                          const arma::rowvec& chi01, const bool doOH);

// Calculates the intermolecular coupling between OH groups for inclusion
// in the system Hamiltonian needed for coupled spectra calculations
extern void InterCouple(arma::rowvec& omega, const arma::mat& xdipole,
                        const arma::rowvec& chi01, const arma::rowvec& muprime,
                        const arma::mat& u, const arma::rowvec& box,
                        const arma::icube& shift, const int numMols,
                        const int numChromos);

// TODO(Zak): update to use Hist and Hist.print()
extern void PrintSpectra(const arma::cx_rowvec& spectra,
                         const std::string& filename, const int corr,
                         const int nzeros, const double deltaT,
                         const double avgFreq);

#endif
