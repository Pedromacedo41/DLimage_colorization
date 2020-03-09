import torch
import numpy as np 

class PriorFactor():
    ''' Class handles prior factor '''
    def __init__(self,alpha,gamma=0,verbose=True,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = torch.as_tensor(np.load(priorFile), dtype= torch.float32)

        # define uniform probability
        self.uni_probs = torch.as_tensor(np.zeros_like(self.prior_probs), dtype= torch.float32)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/torch.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/torch.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/torch.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(torch.min(self.prior_factor),torch.max(self.prior_factor),torch.mean(self.prior_factor),torch.median(self.prior_factor),torch.sum(self.prior_factor*self.prior_probs)))

    '''
    def forward(self,data_ab_quant,axis=1):
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]
    '''

    def compute(self, input):
        q_ind_max_img = torch.argmax(input, dim=1)
        return self.prior_factor[q_ind_max_img]
