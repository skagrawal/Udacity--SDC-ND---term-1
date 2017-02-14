import numpy as np
import cv2
import glob
import pickle


class tracker():
    def __init__(self,window_width,window_height,margin,ym = 1,xm = 1,smooth_factor = 15):
        self.recent_centers=[]
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.ym_per_pix = ym
        self.xm_per_pix = xm
        self.smooth_factor = smooth_factor
        
    def find_window_centroids(self,warped):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        # centroid left and right pairs
        window_centroids = []
        
        window = np.ones(window_width)
        # Take a histogram of the bottom half of the image
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        # One slice
        window_centroids.append((l_center,r_center))
        old_l_center = l_center
        #Looping for remaining slices
        for level in range(1,(int)(warped.shape[0]/window_height)):
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):
                                        int(warped.shape[0]-level*window_height),:], axis=0)
            
            conv_signal = np.convolve(window,image_layer)
            offset = window_width/2
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            l_min_idx = int(max(l_center+offset-margin,0))
            l_max_idx = int(max(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - offset
            
            # Find the best right centroid by using past right center as a reference
            r_min_idx = int(max(r_center+offset-margin,0))
            r_max_idx = int(max(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - offset
            # Append to centroid array
            if abs(old_l_center-l_center)>200:
                l_center = old_l_center
            else:
                old_l_center = l_center
            window_centroids.append((l_center,r_center))
        
        self.recent_centers.append(window_centroids)
        return np.average(self.recent_centers[-self.smooth_factor:],axis=0)
    



