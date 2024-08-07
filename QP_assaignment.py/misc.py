#
# ###############################################################################
# # find out the names of the keys in the vep file
# #############################################################################
# # from scipy.io import loadmat
# #
# # # Load P300.mat
# # p300_mat = loadmat('P300.mat')  # Adjust the path if necessary
# # print("P300.mat keys:", p300_mat.keys())
# #
# # # Print out sample values and lengths for P300.mat
# # if 'eeg' in p300_mat:
# #     print("Length of EEG data in P300.mat:", len(p300_mat['eeg']))
# #     print("Sample EEG values from P300.mat:", p300_mat['eeg'][:5])
# # if 'trigs' in p300_mat:
# #     print("Sample triggers from P300.mat:", p300_mat['trigs'][:5])
# # if 'trig_times' in p300_mat:
# #     print("Sample trigger times from P300.mat:", p300_mat['trig_times'][:5])
# #
# # # Load vep.mat
# # vep_mat = loadmat('vep.mat')  # Adjust the path if necessary
# # print("vep.mat keys:", vep_mat.keys())
# #
# # # Print out sample values and lengths for vep.mat
# # if 'EEG' in vep_mat:
# #     print("Length of EEG data in vep.mat:", len(vep_mat['EEG']))
# #     print("Sample EEG values from vep.mat:", vep_mat['EEG'][:5])
# # if 'stim_times' in vep_mat:
# #     print("Sample stimulus times from vep.mat:", vep_mat['stim_times'][:5])
# #
# # # Additional checks
# # # Checking if trigger times are within the EEG data range
# # if 'eeg' in p300_mat and 'trig_times' in p300_mat:
# #     eeg_length = len(p300_mat['eeg'])
# #     trigger_times = p300_mat['trig_times'][0]  # Assuming trigger times are the first element
# #     if all(0 <= tt < eeg_length for tt in trigger_times):
# #         print("All trigger times from P300.mat are within the EEG data range.")
# #     else:
# #         print("Some trigger times from P300.mat are outside the EEG data range.")
# # ################################################################################
# #
# ############################################################################
# # def mandelbrot(c, max_iter):
# #     z = 0
# #     n = 0
# #     while abs(z) <= 2 and n < max_iter:
# #         z = z*z + c
# #         n += 1
# #     return n
# #
# # width, height = 800, 800
# #
# # re_min, re_max = -2, 2
# # im_min, im_max = -2, 2
# #
# # image = np.zeros((width, height))
# # for x in range(width):
# #     for y in range(height):
# #         c = complex(re_min + (x / width) * (re_max - re_min),
# #                     im_min + (y / height) * (im_max - im_min))
# #         image[x, y] = mandelbrot(c, 256)
# #
# # plt.imshow(image, extent=(re_min, re_max, im_min, im_max))
# # plt.gray()
# # plt.show()
# #####################################################################make a mandelbot set
