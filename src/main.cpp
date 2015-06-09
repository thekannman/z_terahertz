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

// TODO(Zak): The original version had various solvation mode calculations
// included. Those calculations need to be made into their own program.
// TODO(Zak): The original version had printouts of subcontributions
// (perm,ind,ion,etc) as well as their cross contributions. Add those back in.
#include <fftw3.h>
#include <complex>
#include "xdrfile_trr.h"
#include "boost/program_options.hpp"
#include "z_sim_params.hpp"
#include "z_vec.hpp"
#include "z_cx_tcf.hpp"
#include "z_constants.hpp"
#include "z_conversions.hpp"
#include "z_molecule.hpp"
#include "z_subsystem_group.hpp"
#include "z_gromacs.hpp"

namespace po = boost::program_options;
// Units are nm, ps.

int main (int argc, char *argv[]) {
  int st;
  SimParams params;
  int steps_guess = 1000;
  const int kStepsGuessIncrement = 1000;

  double cation_gamma, anion_gamma;
  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("oxygen,O", po::value<std::string>()->default_value("OW"),
     "Group consisting of water oxygens for distance calculations")
    ("water,W", po::value<std::string>()->default_value("SOL"),
     "Group consisting of water molecules for electric field and terahertz"
     "calculations")
    ("solute,S", po::value<std::string>()->default_value("Ion"),
     "Group consisting of charged solutes for electric field calculations")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("max_time,t",
     po::value<double>()->default_value(0.0),
     "Maximum simulation time to use in calculations")
    ("cation_gamma", po::value<double>(&cation_gamma),
     "Cation gamma value for induced dipole calculations")
    ("anion_gamma", po::value<double>(&anion_gamma),
     "Anion gamma value for induced dipole calculations");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());
  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                 params);

  SystemGroup all_atoms(vm["gro"].as<std::string>(), molecules);

  SubsystemGroup *oxygen_group_pointer =
      SubsystemGroup::MakeSubsystemGroup(
          vm["oxygen"].as<std::string>(),
          SelectGroup(groups, vm["oxygen"].as<std::string>()), all_atoms);
  SubsystemGroup &oxygen_group = *oxygen_group_pointer;
  SubsystemGroup *water_group_pointer =
      SubsystemGroup::MakeSubsystemGroup(
          vm["water"].as<std::string>(),
          SelectGroup(groups, vm["water"].as<std::string>()), all_atoms);
  SubsystemGroup &water_group = *water_group_pointer;

  oxygen_group.set_gammas(1.495e-3);

  bool just_water = true;
  std::vector<int> solute_indices;
  std::string solute_group_name;
  if (vm.count("solute")) {
    solute_indices = SelectGroup(groups, vm["solute"].as<std::string>());
    just_water = false;
    assert(vm.count("cation_gamma") > 0);
    assert(vm.count("anion_gamma") > 0);
  }
  solute_group_name = vm.count("solute") ? vm["solute"].as<std::string>() : "";

  SubsystemGroup *solute_group_pointer =
      SubsystemGroup::MakeSubsystemGroup(solute_group_name, solute_indices,
                                         all_atoms);
  SubsystemGroup &solute_group = *solute_group_pointer;
  solute_group.set_ion_gammas(cation_gamma, anion_gamma);

  arma::mat dipole = arma::zeros<arma::mat>(steps_guess, DIMS);
  arma::mat conductivity = arma::zeros<arma::mat>(steps_guess, DIMS);

  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  std::string trr_filename = "prod.trr";
  XDRFILE *xtc_file, *trr_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  trr_file = xdrfile_open(strdup(trr_filename.c_str()), "r");
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_box(box);
  params.set_max_time(vm["max_time"].as<double>());

  int  corr_int = 1;

  oxygen_group.OpenFieldFile();
  solute_group.OpenFieldFile();

  arma::imat nearby_molecules;
  arma::rowvec dx, dipole_contribution;
  rvec *v_in = NULL;
  v_in = new rvec [params.num_atoms()];
  float time, lambda, prec;
  int step = 0;
  double avg_volume;
  for (step = 0; step < params.max_steps(); ++step) {
    if(read_xtc(xtc_file, params.num_atoms(), &st, &time, box_mat, x_in, &prec))
      break;
    if(!just_water) {
      if(read_trr(trr_file, params.num_atoms(), &st, &time, &lambda, box_mat,
                  NULL, v_in, NULL)) {
        break;
      }
    }
    if (step == steps_guess) {
      steps_guess += kStepsGuessIncrement;
      dipole.resize(steps_guess, DIMS);
      conductivity.resize(steps_guess, DIMS);
    }
    params.set_box(box_mat);
    avg_volume += params.volume();

    oxygen_group.set_positions(x_in);
    water_group.set_phase_space(x_in, v_in);
    if (!just_water)
      solute_group.set_phase_space(x_in, v_in);

    oxygen_group.SetElectricField(water_group, params.box());
    if (!just_water)
      oxygen_group.UpdateElectricField(solute_group, params.box());
    if (!oxygen_group.field_check())
      oxygen_group.WriteElectricField();

    solute_group.SetElectricField(water_group, params.box());
    if (!just_water)
      solute_group.UpdateElectricField(solute_group, params.box());
    if (!solute_group.field_check())
      solute_group.WriteElectricField();

    for (int i_atom = 0; i_atom < water_group.size(); ++i_atom) {
      conductivity.row(step) +=
          water_group.velocity(i_atom)*water_group.charge(i_atom);
    }
    for (int i_atom = 0; i_atom < solute_group.size(); ++i_atom) {
      conductivity.row(step) +=
          solute_group.velocity(i_atom)*solute_group.charge(i_atom);
      dipole.row(step) +=
          solute_group.electric_field(i_atom)*solute_group.gamma(i_atom);
    }
    for (int i_atom = 0; i_atom < oxygen_group.size(); ++i_atom) {
      dipole.row(step) +=
          oxygen_group.electric_field(i_atom)*oxygen_group.gamma(i_atom);
    }
  }
  xdrfile_close(xtc_file);
  xdrfile_close(trr_file);

  conductivity *= ENM_TO_D;
  avg_volume *= NM_TO_M*NM_TO_M*NM_TO_M/step;

  int corr_length = 3000;
  int num_corr = (step - 1 - corr_length)/corr_int;
  assert(num_corr > 0);

  CxTCF dipole_tcf(corr_length, params.dt());
  CxTCF conductivity_tcf(corr_length, params.dt());
  CxTCF cross_tcf(corr_length, params.dt());

  dipole_tcf.CorrelateOneDirection(arma::conv_to<arma::cx_mat>::from(dipole));
  dipole_tcf.MultiplyTCF(2.0*M_PI/3.0/C_SPEED*K_COUL*D_TO_CM*D_TO_CM*
                         params.beta()/avg_volume);
  dipole_tcf.FourierPlus();
  dipole_tcf.FreqWeighting(2.0);
  conductivity_tcf.CorrelateOneDirection(
      arma::conv_to<arma::cx_mat>::from(conductivity));
  conductivity_tcf.MultiplyTCF(2.0*M_PI/3.0/C_SPEED*K_COUL*D_TO_CM*D_TO_CM*
                               params.beta()/avg_volume);
  conductivity_tcf.FourierPlus();

  std::string tcf_filename = "thz_tcf.txt";
  std::ofstream tcf_file(tcf_filename.c_str());
  for (int i = 0; i < dipole_tcf.total_length(); ++i) {
      tcf_file << i*dipole_tcf.x_spacing() << " " <<
                  dipole_tcf.tcf(i) + conductivity_tcf.tcf(i) << std::endl;
  }
  tcf_file.close();

  std::string spec_filename = "thz_spec.txt";
  std::ofstream spec_file(spec_filename.c_str());
  for (int i = 0; i < dipole_tcf.total_length(); ++i) {
      spec_file << i*dipole_tcf.freq_spacing_nu() << " " <<
                   dipole_tcf.ft(i) + conductivity_tcf.ft(i) << std::endl;
  }
  spec_file.close();
}
 // main
