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

#include "z_map.hpp"
#include "z_vec.hpp"

double SwitchFunction(const double z, const double u_z, const double dipPos,
                      const double zcenter, const double zboxL,
                      const double rswitch, const double rswitch2,
                      const double rswitch3) {
  double ztemp = z + dipPos*u_z - zcenter;
  if (ztemp>zboxL/2)
    ztemp -= zboxL;
  if (ztemp<-zboxL/2)
    ztemp += zboxL;
  if (ztemp>rswitch)
    return 1.0;
  else if (ztemp<-rswitch)
    return -1.0;
  else
    return (2.0*((3.0*rswitch2*ztemp)-(ztemp*ztemp*ztemp))/(4.0*rswitch3));
}

void ThreeSiteMap(const double eh, const arma::rowvec& u, double& omega01,
                  double& chi01, arma::rowvec& mu01, double& muprime,
                  arma::rowvec& alphaDiag, arma::rowvec& alphaOffDiag,
                  const bool doOH) {
  double omega01_0, omega01_1, omega01_2, chi01_0, chi01_1;
  const double muprime_0 = 0.71116, muprime_1 = 75.591;

  if (doOH) {
    // Not set up yet!!!
    //omega01_0 = 3760.2; omega01_1 = -3541.7; omega01_2 = -152677.0;
    //chi01_0 = 0.19285; chi01_1 = -1.7261E-5;
  } else {
    omega01_0 = 2762.6; omega01_1 = -3640.8; omega01_2 = -56641.0;
    chi01_0 = 0.0880; chi01_1 = -1.105E-5;
  }
  if (eh>=0) {
    omega01  = omega01_0 + omega01_1*eh + omega01_2*eh*eh;
    muprime = muprime_0 + muprime_1*eh;
  } else {
    omega01  = omega01_0 + omega01_1*eh;
    muprime = muprime_0 + muprime_1*eh;
  }
  chi01 = chi01_0 + chi01_1*omega01;
  mu01 = muprime*chi01*u;
  for (int i1=0; i1<DIMS; i1++)
    alphaDiag(i1) = chi01 * (4.6*u(i1)*u(i1)+1.0);
  alphaOffDiag(0) = chi01*4.6*u(1)*u(2);
  alphaOffDiag(1) = chi01*4.6*u(2)*u(0);
  alphaOffDiag(2) = chi01*4.6*u(0)*u(1);
}

void FourSiteMap(const double eh, const arma::rowvec& u, double& omega01,
                 double& chi01, arma::rowvec& mu01, double& muprime,
                 arma::rowvec& alphaDiag, arma::rowvec& alphaOffDiag,
                 const bool doOH) {
  double omega01_0, omega01_1, omega01_2, chi01_0, chi01_1;
  const double muprime_0 = 0.1646, muprime_1 = 11.39, muprime_2 = 63.41;

  if (doOH) {
    omega01_0 = 3760.2; omega01_1 = -3541.7; omega01_2 = -152677.0;
    chi01_0 = 0.19285; chi01_1 = -1.7261E-5;
  } else {
    omega01_0 = 2767.8; omega01_1 = -2630.3; omega01_2 = -102601.0;
    chi01_0 = 0.16593; chi01_1 = -2.0632E-5;
  }
  if (eh>=0) {
    omega01  = omega01_0 + omega01_1*eh + omega01_2*eh*eh;
    muprime = muprime_0 + muprime_1*eh + muprime_2*eh*eh;
  } else {
    omega01  = omega01_0 + omega01_1*eh;
    muprime = muprime_0 + muprime_1*eh;
  }
  chi01 = chi01_0 + chi01_1*omega01;
  mu01 = muprime*chi01*u;
  for (int i1=0; i1<DIMS; i1++)
    alphaDiag(i1) = chi01 * (4.6*u(i1)*u(i1)+1.0);
  alphaOffDiag(0) = chi01*4.6*u(1)*u(2);
  alphaOffDiag(1) = chi01*4.6*u(2)*u(0);
  alphaOffDiag(2) = chi01*4.6*u(0)*u(1);
}

void FourSiteMapIR(const double eh, const arma::rowvec& u, double& omega01,
                   double& chi01, arma::rowvec& mu01, double& muprime,
                   const bool doOH) {
  double omega01_0, omega01_1, omega01_2, chi01_0, chi01_1;
  const double muprime_0 = 0.1646, muprime_1 = 11.39, muprime_2 = 63.41;

  if (doOH) {
    omega01_0 = 3760.2; omega01_1 = -3541.7; omega01_2 = -152677.0;
    chi01_0 = 0.19285; chi01_1 = -1.7261E-5;
  } else {
    omega01_0 = 2767.8; omega01_1 = -2630.3; omega01_2 = -102601.0;
    chi01_0 = 0.16593; chi01_1 = -2.0632E-5;
  }
  if (eh>=0) {
    omega01  = omega01_0 + omega01_1*eh + omega01_2*eh*eh;
    muprime = muprime_0 + muprime_1*eh + muprime_2*eh*eh;
  } else {
    omega01  = omega01_0 + omega01_1*eh;
    muprime = muprime_0 + muprime_1*eh;
  }
  chi01 = chi01_0 + chi01_1*omega01;
  mu01 = muprime*chi01*u;
}

void FourSiteMap2DIR(const double eh, double& omega01, double& omega12,
                     double& chi01, double& chi12, double& mu01,
                     double& mu12, double& muprime, const bool doOH) {
  double omega01_0, omega01_1, omega01_2, omega12_0, omega12_1, omega12_2;
  double chi01_0, chi01_1, chi12_0, chi12_1;
  const double muprime_0 = 0.1646, muprime_1 = 11.39, muprime_2 = 63.41;

  if (doOH) {
    omega01_0 = 3760.2; omega01_1 = -3541.7; omega01_2 = -152677.0;
    chi01_0 = 0.19285; chi01_1 = -1.7261E-5;
    omega12_0 = 3606.0; omega12_1 = -3498.6; omega12_2 = -198715.0;
    chi12_0 = 0.26836; chi12_1 = -2.3788E-5;
  } else {
    omega01_0 = 2767.8; omega01_1 = -2630.3; omega01_2 = -102601.0;
    chi01_0 = 0.16593; chi01_1 = -2.0632E-5;
    omega12_0 = 2673.0; omega12_1 = -1763.5; omega12_2 = -138534.0;
    chi12_0 = 0.23167; chi12_1 = -2.8596E-5;
  }
  if (eh>=0) {
    omega01  = omega01_0 + omega01_1*eh + omega01_2*eh*eh;
    omega12  = omega12_0 + omega12_1*eh + omega12_2*eh*eh;
    muprime = muprime_0 + muprime_1*eh + muprime_2*eh*eh;
  } else {
    omega01  = omega01_0 + omega01_1*eh;
    omega12  = omega12_0 + omega12_1*eh;
    muprime = muprime_0 + muprime_1*eh;
  }
  chi01 = chi01_0 + chi01_1*omega01;
  mu01 = muprime*chi01;
  chi12 = chi12_0 + chi12_1*omega12;
  mu12 = muprime*chi12;
}

void FourSiteMapSFG(const double eh, arma::rowvec u, const double switcher,
                    double& omega01, double& mu01, double& alpha,
                    const double avgFreq) {
  const double omega01_0 = 2767.8, omega01_1 = -2630.3;
  const double omega01_2 = -102601.0, muprime_0 = 0.1646, muprime_1 = 11.39;
  const double muprime_2 = 63.41, chi01_0 = 0.16593, chi01_1 = -2.0632E-5;

  omega01  = omega01_0 + omega01_1*eh + omega01_2*eh*eh;
  const double muprime = muprime_0 + muprime_1*eh + muprime_2*eh*eh;
  const double chi01 = chi01_0 + chi01_1*omega01;
  mu01 = muprime*chi01*u(2)*switcher;
  alpha = chi01*(4.6*u(0)*u(0)+1.0) + chi01*(4.6*u(1)*u(1)+1.0);
  omega01 = omega01 - avgFreq;
}

double IntraCouple(const double eh1, const double eh2, const double omega011,
                   const double omega012, const arma::rowvec& chi01,
                   const bool doOH) {
  double mom01_0, mom01_1;
  arma::rowvec mom01(2);
  if (doOH) {
    mom01_0 = 1.6466; mom01_1 = 5.7692e-4;
  } else {
    mom01_0 = 2.0475; mom01_1 = 8.9108e-4;
  }
  mom01(0) = mom01_0 + mom01_1*omega011;
  mom01(1) = mom01_0 + mom01_1*omega012;

  return ((-1361.0 + 27165.0*(eh1+eh2))*chi01(0)*chi01(1) -
          1.887*mom01(0)*mom01(1));
}

void InterCouple(arma::rowvec& omega, const arma::mat& xdipole,
                 const arma::rowvec& chi01, const arma::rowvec& muprime,
                 const arma::mat& u, const arma::rowvec& box,
                 const arma::icube& shift, const int numMols,
                 const int numChromos) {
  arma::rowvec xij(3), n(3);
  for(int i=0; i<numMols; i++) {
    const int i2 = 2*i;
    const int i21 = i2+1;
    for (int j=0; j<i; j++) {
      const int j2 = 2*j;
      const int j21 = j2+1;
      xij = UseDx(xdipole.row(i2), xdipole.row(j2), box, shift.tube(i,j));
      double rij = norm(xij);
      //Seems like I'm missing an assignment of n here so added the one below 4/3/2015
      n = xij/rij;
      double udot = dot(u.row(i2),u.row(j2));
      double undot1 = dot(u.row(i2),n);
      double undot2 = dot(u.row(j2),n);
      omega(numChromos*i2+j2) = omega(numChromos*j2+i2) =
          (muprime(i2)*muprime(j2)*(udot-3.0*undot1*undot2)/rij/rij/rij)*
          chi01(i2)*chi01(j2)*32.52278462;

      xij = UseDx(xdipole.row(i2), xdipole.row(j21), box, shift.tube(i,j));
      rij = norm(xij);
      n = xij/rij;
      udot = dot(u.row(i2),u.row(j21));
      undot1 = dot(u.row(i2),n);
      undot2 = dot(u.row(j21),n);
      omega(numChromos*i2+j21) = omega(numChromos*j21+i2) =
          (muprime(i2)*muprime(j21)*(udot-3.0*undot1*undot2)/rij/rij/rij)*
          chi01(i2)*chi01(j21)*32.52278462;

      xij = UseDx(xdipole.row(i21), xdipole.row(j2), box, shift.tube(i,j));
      rij = norm(xij);
      n = xij/rij;
      udot = dot(u.row(i21),u.row(j2));
      undot1 = dot(u.row(i21),n);
      undot2 = dot(u.row(j2),n);
      omega(numChromos*i21+j2) = omega(numChromos*j2+i21) =
          (muprime(i21)*muprime(j2)*(udot-3.0*undot1*undot2)/rij/rij/rij)*
          chi01(i21)*chi01(j2)*32.52278462;

      xij = UseDx(xdipole.row(i21), xdipole.row(j21), box, shift.tube(i,j));
      rij = norm(xij);
      n = xij/rij;
      udot = dot(u.row(i21),u.row(j21));
      undot1 = dot(u.row(i21),n);
      undot2 = dot(u.row(j21),n);
      omega(numChromos*i21+j21) = omega(numChromos*j21+i21) =
          (muprime(i21)*muprime(j21)*(udot-3.0*undot1*undot2)/rij/rij/rij)*
          chi01(i21)*chi01(j21)*32.52278462;
    }
  }
}

void PrintSpectra(const arma::cx_rowvec& spectra, const std::string& filename,
                  const int corr, const int nzeros, const double deltaT,
                  const double avgFreq) {
  std::ofstream file;
  file.open(filename.c_str());
  double factor = 1.0/2.0/C_SPEED/(deltaT*static_cast<double>(corr+nzeros));

  int imax = 2*(corr+nzeros)-2;
  for (int i=corr+nzeros; i<imax; i++) {
    file << std::fixed <<
            static_cast<double>(i+2-2*(corr+nzeros))*factor+avgFreq << " ";
    file << std::scientific << real(spectra(i)) << std::endl;
  }
  imax = corr+nzeros;
  for (int i=0; i<imax; i++) {
    file << std::fixed << static_cast<double>(i)*factor + avgFreq << " ";
    file << std::scientific << real(spectra(i)) << std::endl;
  }
  file.close();
}
