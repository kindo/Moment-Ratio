
class MomentRatio(): 
    
    """
    module: Moment ratio 
    Methods:
        - __init__: input sample as a numpy array
        
        - pwm_br_2: method to compute the probability weighted moment 
            ref: https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2001WR900014
            
        - lmoments: method to compute lmoments: 
            Hosking, J. R. M. (1990). 
            L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics. 
            Journal of the Royal Statistical Society. Series B (Methodological), 52(1), 
            105–124. http://www.jstor.org/stable/2345653
            
        - lmoments_ratios: method to compute the lmoments ratios of order r
        
        - l_cv: method to compute the l-coefficient of variation 
        
        - fit: method to populate LM_statistic: with the lmoment and lmoment ratio of the data
        
        - plot_MRD method to plot the sample in the MRD see:
        Hosking, J. R. M., and J. R. Wallis. 1997. 
        Regional Frequency Analysis: An Approach Based on L-Moments. Cambridge University Press, Cambridge, p.
        
    """

    
    def __init__(self, data = None, LM_statistic = None):
        self.data = data
        self.LM_statistic = LM_statistic
        
    def __repr__(self):
        return '<class MomentRatio>'
        
    
    def pwm_br_2(self, r=1):
        
        
        """"
        Method to compute the probability weighted moment (Rasmussen, 2001)
        
        ========================================================================
        ref1: 
        Rasmussen, P. F. (2001), Generalized probability weighted moments: 
        Application to the generalized Pareto Distribution, 
        Water Resour. Res., 37( 6), 1745– 1751, 
        doi:10.1029/2001WR900014.
            
        """"
        
        from scipy.special import comb
        import numpy as np 
        
        sample = np.sort(self.data)
        n = len(sample)
        br =  np.array([(comb(i - 1, r, exact=True)/comb(n-1, r, exact=True))*sample[i-1] for i in range(1, n+1)]).sum()/n
        return br

    
    def lmoments(self, l):
        
        
        """
        lmoments: method to compute lmoments (Hosking, 1990) 
        
        ==================================================================================================
        ref1:
        Hosking, J. R. M. (1990). 
        L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics. 
        Journal of the Royal Statistical Society. Series B (Methodological), 52(1), 
        105–124. http://www.jstor.org/stable/2345653
        
        """
        
        
        if l == 1:
            return self.pwm_br_2(0)
        elif l==2:
            return 2*self.pwm_br_2(1) - self.pwm_br_2(0)
        elif l == 3:
            return 6*self.pwm_br_2(2) - 6*self.pwm_br_2(1) + self.pwm_br_2(0)
        elif l == 4:
            return 20*self.pwm_br_2(3) - 30*self.pwm_br_2(2) + 12*self.pwm_br_2(1) - self.pwm_br_2(0)
        
    def lmoments_ratios(self, r):
        
        """
        
        lmoments_ratios: method to compute the lmoments ratios of order r
        
        """
        
        return self.lmoments(r)/self.lmoments(2)
    
    def l_cv(self):
        
        """
        l_cv: method to compute the l-coefficient of variation 
        
        """
        
        
        return self.lmoments(2)/self.lmoments(1)
    
    def fit(self):
        
        """
        fit: method to populate LM_statistic: with the lmoment and lmoment ratio of the data
        
        """
        
        
        
        import pandas as pd
        LM_statistic_ = {}
        
        LM_statistic_["L1"] = self.lmoments(1)
        LM_statistic_["L2"] = self.lmoments(2)
        LM_statistic_["L3"] = self.lmoments(3)
        LM_statistic_["L4"] = self.lmoments(4)
        LM_statistic_["L5"] = self.lmoments(5)
        LM_statistic_["L3/L2"] = self.lmoments_ratios(3)
        LM_statistic_["L4/L2"] = self.lmoments_ratios(4)
        LM_statistic_["L2/L1"] = self.l_cv()
        
        
        self.LM_statistic = pd.Series(LM_statistic_, name='LMoment_statistics')
        
        
    def plot_MRD(self):
        
        """
        Method to plot the sample lmoment ratios (skewness and kurtosis) in the MRD 
        
        ==============================================================
        see:
        Hosking, J. R. M., and J. R. Wallis. 1997. 
        Regional Frequency Analysis: An Approach Based on L-Moments. 
        Cambridge University Press, Cambridge, p.
        
        
        """
        
        from ipywidgets import interact
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    
        def plotting_MRD(xmin=-1, xmax=1, ymin=-0.24, ymax=1, distributions=True):
    
            
            
            fig, ax = plt.subplots()
            
            if distributions:
                distributionLMR = pd.read_csv("Distributions_LMR.csv", index_col=False)
                colname = distributionLMR.columns.values


                for d in range(0, distributionLMR.shape[1], 4):
                    L3I = distributionLMR.iloc[:, d]
                    L3S = distributionLMR.iloc[:, d+1]
                    L4I = distributionLMR.iloc[:, d+2]
                    L4S = distributionLMR.iloc[:, d+3]
                    
                    if colname[d][:-4] == "EV1":
                        ax.scatter(L3I, L4I, label="EV1")
                        continue

                    if np.all(L3S.isna()):
                        sns.lineplot(x = L3I, y=L4I, label=colname[d][:-4], ax=ax)
                    else:
                        sns.lineplot(x=L3I, y=L4I, label=f"{colname[d][:-4]} Inf", ax=ax)
                        sns.lineplot(x=L3S, y=L4S, label = f"{colname[d][:-4]} Sup", ax=ax)

            if isinstance(self.LM_statistic, pd.Series) :
                ax.scatter(x = 'L3/L2',  y= 'L4/L2', data = self.LM_statistic, label="data")
            
            
            ax.set(xlabel="L3/L2 (Skewness)", ylabel="L4/L2 (Kurtosis)", xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.grid()
            ax.legend(bbox_to_anchor=(1.01, 1.01))


        interact(plotting_MRD, xmin = (-1.1, 1), xmax = (-1.1, 1), ymin=(-0.24, 1), ymax=(-0.24, 1), distributions=True)
            
            
