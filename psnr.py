import os

def calculate_average_metrics(base_directory):
    total_psnr, total_ssim, total_lpips_alex, total_lpips_vgg, total_parameters, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    count = 0

    # Traverse the base directory and subdirectories
    for root, dirs, files in os.walk(base_directory):
        for dir in dirs:
            exp_path = os.path.join(root, dir, "imgs_test_all")  
            mean_file_path = os.path.join(exp_path, "mean.txt")
            time_file_path = os.path.join(exp_path, "time.txt")
            
            # Check if the mean.txt file exists
            if os.path.isfile(mean_file_path):
                with open(mean_file_path, 'r') as f:
                    try:
                        # Read the metrics
                        lines = f.readlines()
                        psnr = float(lines[0].strip())
                        ssim = float(lines[1].strip())
                        l_a = float(lines[2].strip())
                        l_v = float(lines[3].strip())
                        n_params = float(lines[4].strip())

                        # Accumulate metrics
                        total_psnr += psnr
                        total_ssim += ssim
                        total_lpips_alex += l_a
                        total_lpips_vgg += l_v
                        total_parameters += n_params
                        count += 1
                    except (IndexError, ValueError):
                        print(f"Error reading metrics file: {mean_file_path}")

            # Check if the time.txt file exists
            if os.path.isfile(time_file_path):
                with open(time_file_path, 'r') as f:
                    try:
                        # Read the time value
                        time_value = float(f.readline().strip())
                        total_time += time_value
                    except (ValueError, IndexError):
                        print(f"Error reading time file: {time_file_path}")
    
    # Compute averages
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips_alex = total_lpips_alex / count
        avg_lpips_vgg = total_lpips_vgg / count
        avg_parameters = total_parameters / count
        avg_time = total_time / count

        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS (AlexNet): {avg_lpips_alex:.4f}")
        print(f"Average LPIPS (VGG): {avg_lpips_vgg:.4f}")
        print(f"Average Parameters: {avg_parameters:.2e}")
        print(f"Average Time: {avg_time:.2f} seconds")
    else:
        print("No valid metrics or time data found.")

# Call the function
# base_directory = "logsfactorfields"
base_directory = "logsfactorfieldsSH"
calculate_average_metrics(base_directory)