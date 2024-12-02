import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Geometry import Point3D
import py3Dmol

class SimpleMolecularDynamics:
    def __init__(self, temperature=300, step_size=0.05):
        self.kB = 0.001987
        self.temperature = temperature
        self.beta = 1 / (self.kB * temperature)
        self.step_size = step_size
        
    def calculate_energy(self, mol):
        try:
            # MMFF94 Energy
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff is None:
                return float('inf')
            return ff.CalcEnergy()
        except:
            return float('inf')
            
    def run_metropolis(self, mol, n_steps=1000):
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            print("Failed to generate 3D conformation")
            return None
            
        energies = []
        positions = []
        accepted_moves = 0
        current_energy = self.calculate_energy(mol)
        
        for step in range(n_steps):
            new_mol = Chem.Mol(mol)
            conf = new_mol.GetConformer()
            
            # Random perturbation
            for i in range(new_mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                new_pos = Point3D(
                    pos.x + np.random.normal(0, self.step_size),
                    pos.y + np.random.normal(0, self.step_size),
                    pos.z + np.random.normal(0, self.step_size)
                )
                conf.SetAtomPosition(i, new_pos)
            
            new_energy = self.calculate_energy(new_mol)
            
            # Metropolis criterion
            delta_E = new_energy - current_energy
            if delta_E < 0 or np.random.random() < np.exp(-self.beta * delta_E):
                mol = new_mol
                current_energy = new_energy
                accepted_moves += 1
            
            energies.append(current_energy)
            positions.append(self.get_positions(mol))
            
        return {
            'energies': energies,
            'positions': positions,
            'acceptance_rate': accepted_moves / n_steps,
            'final_energy': current_energy,
            'final_structure': mol
        }
    
    def get_positions(self, mol):
        conf = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        return positions

class DTISimulator:
    def __init__(self):
        self.md_sim = SimpleMolecularDynamics()
    
    def calculate_binding_score(self, smiles, protein_seq):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Failed to parse SMILES: {smiles}")
                return None
                
            # Calculate molecular descriptors
            descriptors = {
                'MW': Descriptors.ExactMolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'ArRings': Descriptors.NumAromaticRings(mol),
                'Complexity': Descriptors.BertzCT(mol)
            }
            
            # Run simulation
            sim_results = self.md_sim.run_metropolis(mol, n_steps=100)
            if sim_results is None:
                return None
            
            # Calculate binding score using molecular descriptors and simulation results
            score = (
                -0.1 * sim_results['final_energy'] +  # Energy contribution
                -0.05 * descriptors['MW'] +           # Size penalty
                -0.2 * abs(descriptors['LogP']) +     # Hydrophobicity term
                0.1 * (descriptors['HBD'] + descriptors['HBA']) +  # H-bonding capacity
                -0.01 * descriptors['TPSA'] +         # Surface area term
                -0.1 * descriptors['RotBonds'] +      # Flexibility penalty
                0.2 * descriptors['ArRings'] +        # Aromatic interactions
                -0.001 * descriptors['Complexity']    # Complexity penalty
            )
            
            return {
                'binding_score': score,
                'molecular_properties': descriptors,
                'simulation_results': {
                    'energies': sim_results['energies'],
                    'final_energy': sim_results['final_energy'],
                    'acceptance_rate': sim_results['acceptance_rate'],
                    'final_structure': sim_results['final_structure']
                }
            }
            
        except Exception as e:
            print(f"Error in binding score calculation: {e}")
            return None


class DTIAnalyzer:
    def __init__(self):
        self.binding_threshold = -10  # Define threshold for significant binding
        
    def analyze_simulation_results(self, results):
        """Analyze simulation results and provide detailed metrics"""
        if not results:
            return None
            
        analysis = {
            'binding_strength': self.classify_binding_strength(results['binding_score']),
            'molecular_analysis': self.analyze_molecular_properties(results['molecular_properties']),
            'energy_profile': {
                'binding_score': results['binding_score'],
                'normalized_score': self.normalize_binding_score(results['binding_score']),
            }
        }
        
        return analysis
    
    def classify_binding_strength(self, binding_score):
        """Classify binding strength based on score"""
        if binding_score < -20:
            return "Strong Binding"
        elif binding_score < -10:
            return "Moderate Binding"
        elif binding_score < -5:
            return "Weak Binding"
        else:
            return "No Significant Binding"
    
    def analyze_molecular_properties(self, properties):
        """Analyze molecular properties against typical drug-like ranges"""
        analysis = {}
        
        # Lipinski's Rule of Five analysis
        lipinski_violations = 0
        if properties['MW'] > 500: lipinski_violations += 1
        if properties['LogP'] > 5: lipinski_violations += 1
        if properties['HBD'] > 5: lipinski_violations += 1
        if properties['HBA'] > 10: lipinski_violations += 1
        
        analysis['lipinski_violations'] = lipinski_violations
        analysis['drug_likeness'] = "Good" if lipinski_violations <= 1 else "Poor"
        
        # Property-specific analysis
        analysis['mw_category'] = "Optimal" if 160 < properties['MW'] < 500 else "Non-optimal"
        analysis['logp_category'] = "Optimal" if -0.4 < properties['LogP'] < 5.6 else "Non-optimal"
        
        return analysis
    
    def normalize_binding_score(self, score):
        """Normalize binding score to 0-1 range"""
        # Assuming typical binding scores range from 0 to -30
        return max(0, min(1, -score/30))
    
    def visualize_molecule_2d(self, smiles):
        """Generate 2D molecule visualization"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol)
        return img
    
    def visualize_molecule_3d(self, mol):
        """Generate 3D molecule visualization using py3Dmol"""
        mb = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(mb, "mol")
        viewer.setStyle({'stick':{}})
        viewer.zoomTo()
        return viewer
    
    def plot_energy_trajectory(self, energies):
        """Plot energy trajectory during simulation"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=energies,
            mode='lines',
            name='Energy'
        ))
        fig.update_layout(
            title='Energy Trajectory During Simulation',
            xaxis_title='Simulation Step',
            yaxis_title='Energy (kcal/mol)'
        )
        return fig
    
    def generate_report(self, smiles, protein_seq, results):
        """Generate comprehensive analysis report"""
        analysis = self.analyze_simulation_results(results)
        
        report = {
            'summary': {
                'binding_strength': analysis['binding_strength'],
                'drug_likeness': analysis['molecular_analysis']['drug_likeness'],
                'binding_score': results['binding_score']
            },
            'molecular_properties': results['molecular_properties'],
            'analysis': analysis,
            'visualizations': {
                'mol_2d': self.visualize_molecule_2d(smiles),
                'mol_3d': self.visualize_molecule_3d(results['simulation_results']['final_structure']),
                'energy_plot': self.plot_energy_trajectory(results['simulation_results']['energies'])
            }
        }
        
        return report

# Example usage
if __name__ == "__main__":
    from simulator import DTISimulator
    
    # Initialize simulator and analyzer
    simulator = DTISimulator()
    analyzer = DTIAnalyzer()
    
    # Test data
    test_smiles = "CC1=CC=C(C=C1)C2=CN=C(N=C2N)N"
    test_protein = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTH"
    
    # Run simulation
    results = simulator.calculate_binding_score(test_smiles, test_protein)
    
    # Generate and display report
    if results:
        report = analyzer.generate_report(test_smiles, test_protein, results)
        
        print("\n=== DTI Analysis Report ===")
        print(f"\nBinding Strength: {report['summary']['binding_strength']}")
        print(f"Drug-likeness: {report['summary']['drug_likeness']}")
        print(f"Binding Score: {report['summary']['binding_score']:.2f}")
        
        print("\nMolecular Properties Analysis:")
        for prop, value in report['analysis']['molecular_analysis'].items():
            print(f"{prop}: {value}")
        
        print("\nVisualization objects generated:")
        print("- 2D structure")
        print("- 3D structure")
        print("- Energy trajectory plot")