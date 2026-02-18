from deconvolutioner import XRD_Deconvolution_Process
import sys
sys.dont_write_bytecode = True

pkl_file = "rs_profile/martin_senai_rs_profile.pkl" 

try:
    processor = XRD_Deconvolution_Process(
        pkl_filename=pkl_file,
        beam_diameter_mm=0.5,   
        overlap_ratio=0.50,     
        noise_std_dev=15.0      
    )
    
    processor.run_full_process()
    processor.plot_results()
    
except FileNotFoundError:
    print(f"Error: Please run the previous code to generate '{pkl_file}' first.")