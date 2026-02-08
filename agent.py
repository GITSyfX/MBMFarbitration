import numpy as np
from scipy.special import softmax

eps_, max_ = 1e-12, 1e12
sigmoid = lambda x: 1.0 / (1.0 + clip_exp(-x))
logit   = lambda p: np.log(p) - np.log1p(-p)

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    y = np.exp(x)
    return np.where(y > 1e-11, y, 0)

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = {k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

class RA():   
    name = 'Random Agent'
    bnds=[(0,.5),(0,5)] 
    pbnds= [(.1,.5),(.1,2)]  
    p_name   = ['alpha', 'beta']
    n_params = len(bnds) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x)]  
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_))]

    def __init__(self,env,params):
        self.env = env 
        self._init_mem()
        self._init_critic()
        self._load_params(params)
    
    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0]
        self.beta  = params[1]

    def _init_critic(self):
        self.Q = np.full((self.env.nS, self.env.nA),0.5)
    # ----------- decision ----------- #
    def policy(self,s): 
        q = self.Q[s, :]
        return q
    def eval_act(self, s, a):
        self.prob = self.Q[s, :] 
        return self.prob[int(a)]
    def _init_mem(self):
        self.mem = simpleBuffer()
    # ----------- learning ----------- #
    def learn(self):
        self.learn_critic()
    def learn_critic(self):
        self.Q = self.Q   
    def bw_update(self,g): 
        self.Q = self.Q

class MF():
    name = 'Model Free'
    bnds = [(0,1),(0,5)] #边界
    pbnds = [(.1,.5),(.1,2)] #采样边界
    p_name   = ['alpha', 'beta']  #参数名
    n_params = len(p_name) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x)]  
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_))] 

    gamma = 1 
    
    def __init__(self,env,params):
        self.env = env 
        self.gamma = MF.gamma
        self._init_mem()
        self._init_critic()
        self._load_params(params)

    def _init_mem(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0] # learning rate 
        self.beta  = params[1] # inverse temperature 

    def _init_critic(self):
        self.Q = np.zeros([self.env.nS, self.env.nA])
    # ----------- decision ----------- #

    def policy(self, s):
        q = self.Q[s,:]
        return softmax(self.beta*q)

    def eval_act(self, s, a):
        '''Evaluate the probability of given state and action
        '''
        logit = self.Q[s, :] 
        prob  = softmax(self.beta*logit)
        #print(logit)
        return prob[int(a)]
    
        # ----------- learning ----------- #
    
    def learn(self):
        s, a, s_next,a_next, r, done = self.mem.sample(
                        's','a','s_next','a_next','r','done')
        
        if done != True:  
            self.RPE = r + self.gamma*self.Q[s_next,a_next]-self.Q[s,a]
        else:
            self.RPE = r - self.Q[s,a]
        # Q-update
        self.Q_old = self.Q[s,a]
        self.Q[s,a] = self.Q[s,a]+self.alpha*self.RPE

        #print(RPE, s, a, r, self.alpha, self.Q_old, self.Q[s, a])

        return self.RPE,self.Q_old,self.Q
    
    def bw_update(self,g):
        return self.Q

class MB_raw():
    name = 'Model Based raw'
    bnds = [(0,1),(0,5)] #边界
    pbnds = [(.1,.5),(.1,2)] #采样边界
    p_name   = ['alpha', 'beta']  #参数名
    n_params = len(p_name) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x)]  
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_))] 
    
    def __init__(self,env,params):
        self.env = env 
        self._init_mem()
        self._init_env_model()
        self._init_critic()
        self._load_params(params)
    
    def _init_mem(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0]
        self.beta  = params[1]
    
    def _init_env_model(self):
        self.T_bel = np.zeros(
            [self.env.nS, self.env.nA, self.env.nS] #(s,a,s')
            ) 
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                self.T_bel[s,a,:] = 1/self.env.nS
        
    def _init_critic(self):
        self.Q = np.zeros([self.env.nS, self.env.nA])

    # ----------- decision ----------- #

    def policy(self, s):
        q = self.Q[s,:]
        return softmax(self.beta*q)

    def eval_act(self, s, a):
        '''Evaluate the probability of given state and action
        '''
        logit = self.Q[s, :] 
        prob  = softmax(self.beta*logit)
        return prob[int(a)]
    
    # ----------- learning ----------- #
    def learn(self):

        s, a, s_next, r, g = self.mem.sample(
                        's','a','s_next','r','g')
        
        self.env.set_reward(g)
        
        if s < 5: 
            self.SPE = 1-self.T_bel[s,a,s_next]
            # T-update (increase T(s,a,s') and decrease T(s,a,-) to ensure the sum=1
            self.T_bel[s,a,s_next]= self.T_bel[s,a,s_next] + self.alpha * self.SPE
            array_rest = np.where(np.arange(self.env.nS) != s_next)[0] #the rest of the states
            for j in array_rest:
                self.T_bel[s,a,j] = self.T_bel[s,a,j]*(1-self.alpha) #SPE = 0-self.T


            # compute the prediction error 
            Q_sum=0
            Q_sum=Q_sum+self.T_bel[s,a,s_next]*(r+max(self.Q[s_next,:])) # for (s,a,s')
            
            for j in array_rest: # for the rest
                Q_sum=Q_sum+self.T_bel[s,a,j]*(self.env.R[j]+max(self.Q[j,:]))
                
            self.Q_old = self.Q[s,a]
            self.Q[s,a] = Q_sum
        return self.SPE,self.Q_old,self.Q
    
    def bw_update(self,g): #update all Q(s) when switch reward target
        self.env.set_reward(g)

        Q_bwd_before = self.Q
        for i in range(max(self.env.level),0,-1):
            state_ind_set = np.where(self.env.level==i)
            for l in range(len(state_ind_set)):
                current_S = state_ind_set[l]
                for current_A in range(self.env.nA):
                    tmp_sum=0
                    for j in range(self.env.nS): 
                        tmp_sum = tmp_sum + self.T_bel[current_S,current_A,j] * \
                                (self.env.R[j]+max(self.Q[j,:]))
                    self.Q[current_S,current_A]=tmp_sum
        Q_bwd_after = self.Q
        self.dQ = Q_bwd_after - Q_bwd_before
        self.dQ_bwd_energy = np.sqrt(np.sum(np.sum((self.dQ)**2)))
        self.dQ_mean_energy = np.mean(np.mean(self.dQ))
        return self.Q

class MB():
    name = 'Model Based'
    bnds = [(0,1),(0,5)] #边界
    pbnds = [(.1,.5),(.1,2)] #采样边界
    p_name   = ['alpha', 'beta']  #参数名
    n_params = len(p_name) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x)]  
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_))] 


    def __init__(self,env,params):
        self.env = env 
        self._init_mem()
        self._init_env_model()
        self._init_critic()
        self._load_params(params)
    
    def _init_mem(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0]
        self.beta  = params[1]
    
    def _init_env_model(self):
        self.T_bel = np.zeros(
            [self.env.nS, self.env.nA, self.env.nS] #(s,a,s')
            ) 
        self.T = {0:[[1,2], [6,7], [7,8], [6,5], [6,8]],
                  1:[[3,4], [7,8], [6,8], [5,8], [8,5]]}

        for a in range(self.env.nA):
            for s in range(len(self.T[a])):
                self.T_bel[s,a,self.T[a][s]]=[0.5,1-0.5]        
        
    def _init_critic(self):
        self.Q = np.zeros([self.env.nS, self.env.nA])

    # ----------- decision ----------- #

    def policy(self, s):
        q = self.Q[s,:]
        return softmax(self.beta*q)

    def eval_act(self, s, a):
        '''Evaluate the probability of given state and action
        '''
        logit = self.Q[s, :] 
        prob  = softmax(self.beta*logit)
        return prob[int(a)]
    
    # ----------- learning ----------- #
    def learn(self):

        s, a, s_next, r, g = self.mem.sample(
                        's','a','s_next','r','g')
        
        self.env.set_reward(g)

        if s < 5: 
            self.SPE=1-self.T_bel[s,a,s_next]
            # T-update (increase T(s,a,s') and decrease T(s,a,-) to ensure the sum=1
            self.T_bel[s,a,s_next]= self.T_bel[s,a,s_next] + self.alpha * self.SPE
            s_unchosen = [elem for elem in self.T[a][s] if elem != s_next] # the rest of the states
            for j in s_unchosen:
                self.T_bel[s,a,j] = self.T_bel[s,a,j]*(1-self.alpha) #SPE = 0-self.T

            # Q update
            Q_sum=0
            Q_sum=Q_sum+self.T_bel[s,a,s_next]*(r+max(self.Q[s_next,:])) # for (s,a,s')
            
            for j in s_unchosen: # for the rest
                Q_sum=Q_sum+self.T_bel[s,a,j]*(self.env.R[j]+max(self.Q[j,:]))
                
            self.Q_old = self.Q[s,a] 
            self.Q[s,a]= Q_sum

            #print(SPE, s, a, r, self.alpha, self.Q_old, self.Q[s, a])
        return self.SPE,self.Q_old,self.Q
    
    def bw_update(self,g): #update all Q(s) when switch reward target
        self.env.set_reward(g)

        Q_bwd_before = self.Q
        for i in range(max(self.env.level),0,-1):
            state_ind_set = np.where(self.env.level==i)
            for l in range(len(state_ind_set)):
                current_S = state_ind_set[l]
                for current_A in range(self.env.nA):
                    tmp_sum=0
                    for j in range(self.env.nS): 
                        tmp_sum = tmp_sum + self.T_bel[current_S,current_A,j] * \
                                (self.env.R[j]+max(self.Q[j,:]))
                    self.Q[current_S,current_A]=tmp_sum
        Q_bwd_after = self.Q
        self.dQ = Q_bwd_after - Q_bwd_before
        self.dQ_bwd_energy = np.sqrt(np.sum(np.sum((self.dQ)**2)))
        self.dQ_mean_energy = np.mean(np.mean(self.dQ))
        return self.Q
    
class MDT:
    name = 'MixedArb-Dynamic'
    bnds = [(1e-3, 1), (0, 1), (0.02, 10), (0.02, 10), (0, 1), (0, 5)]
    pbnds = [(0.3, 0.7), (0.1, 0.35), (0.02, 5), (0.02, 5), (0.1, 0.5), (0.1, 2)]
    p_name = ['w','eta','A_F2B','A_B2F','alpha','beta'] #参数名
    n_params = len(bnds) 

    p_trans = [
        
        lambda x: 1e-3 + (1 - 1e-3) * sigmoid(x),     # (1e-3, 1)
        lambda x: 0  + (1 - 0.0)  * sigmoid(x),     # (0, 1)
        lambda x: 0.02  + (10 - 0.02) * sigmoid(x),     # (0.02, 10)
        lambda x: 0.02 + (10 - 0.02)* sigmoid(x),     # (0.02, 10)
        lambda x: 0.0  + (1 - 0.0)  * sigmoid(x),     # (0, 1)
        lambda x: 0.0  + (5 - 0.0)  * sigmoid(x),     # (0, 5)
    ]
    
    p_links = [
        lambda y: logit(np.clip((y - 1e-3) / (1 - 1e-3), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0 ) / (1 - 0.0 ), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.02 ) / (10 - 0.02), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.02) / (10 - 0.02), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0 ) / (1 - 0.0 ), eps_, 1 - eps_)),
        lambda y: logit(np.clip((y - 0.0 ) / (5 - 0.0 ), eps_, 1 - eps_)),
    ]
    
    def __init__(self, env,params):
        self.env = env
        self._load_params(params)
        self._load_agents()
        self._init_mem()
        self._init_critic()
        self._init_PEest()
        self._init_arbitration()
        self._init_Qinteg()

    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.w = params[0] # PE tolerance
        self.eta = params[1] # learning rate of absolute PE estimation 
        self.A_F2B = params[2]
        self.A_B2F = params[3]
        self.alpha = params[4]
        self.beta = params[5]

    def _load_agents(self):
        self.agent_MF = MF(self.env,params=[self.alpha,self.beta])
        self.agent_MB = MB(self.env,params=[self.alpha,self.beta])

    def _init_mem(self):
        self.mem = simpleBuffer()
        
    def _init_critic(self):
        self.Q = np.zeros([self.env.nS, self.env.nA])
            
    def _init_PEest(self):

        '''Beyesian Reliability estimation'''
        self.K=3; # trichonomy of PE
        self.M=19; # memory size = 10
        self.M_half=6; # half-life
        self.M_current_MB=0; # MF accumulated events. cannot exceed T.
    
        # PE history 
        # row: first index = smallest PE -> last index - largest PE
        # column: first index = most recent -> last index = most past
        self.MB_PE_history = np.zeros((self.K,self.M))
        self.MB_PE_num = np.zeros((self.K,1))   

        #self.discount_mat = np.exp((-1)*log(2)*([0:1:self.T-1]/(self.T_half-1))); #prepare discount mat: 1xT
        #self.discount_mat = np.ones((1,self.M)); # no discount
    
        '''Pearce-Hall associability'''
        # Bayesian part - MF RPE estimator
        self.MF_absPEestimate = 0.0

        # Bayesian part - mean,var,inv_Fano
        self.MB_mean = 1/3*np.ones((self.K,1))
        self.MF_mean = 1/3*np.ones((self.K,1))

        self.MB_var = (2/((3^2)*4))*np.ones((self.K,1))  
        self.MF_var = (2/((3^2)*4))*np.ones((self.K,1))
        
        self.MB_inv_Fano = (self.MB_mean)/self.MB_var
        self.MF_inv_Fano = (self.MF_mean)/self.MF_var

    def _init_arbitration(self):
        # weights for {-PE(to be unlearned), 0PE, +PE(to be learned)}
        self.ind_active_model = 1
        self.time_step = 1

        self.B_B2F = np.log(np.maximum(self.A_B2F / 0.01 - 1, 1e-10))
        self.B_F2B = np.log(np.maximum(self.A_F2B / 0.01 - 1, 1e-10))

        if self.ind_active_model == 1:
            self.MB_prob_prev=0.7   
        else:
            self.MB_prob_prev=0.3

        self.MF_prob_prev = 1-self.MB_prob_prev
        self.MB_prob = self.MB_prob_prev
        self.MF_prob = self.MF_prob_prev

        if self.ind_active_model == 1:
            self.num_MB_chosen=1
            self.num_MF_chosen=0
        else:
            self.num_MB_chosen=0
            self.num_MF_chosen=1


    def _init_Qinteg(self):
        #Q-integration part
        self.p = 1 # 1:expectation, 1e1:winner-take-all
        # set the first transition rate to be the equilibrium rate
        if self.B_F2B != self.B_B2F:
            self.inv_Fano_equilibrium = np.log(self.A_F2B/self.A_B2F)/(self.B_F2B-self.B_B2F) # applied only for the unnormalized case
            # self.inv_Fano_equilibrium=.5;
            self.transition_rateB2F_prev = self.A_B2F*np.exp((-1)*self.B_B2F*self.inv_Fano_equilibrium)
            self.transition_rateF2B_prev = self.transition_rateB2F_prev
        else:
            # self.inv_Fano_equilibrium=150;
            self.inv_Fano_equilibrium =.1
            # self.transition_rateB2F_prev=self.A_B2F*exp((-1)*self.B_B2F*self.inv_Fano_equilibrium);
            self.transition_rateB2F_prev = self.inv_Fano_equilibrium
            self.transition_rateF2B_prev = self.transition_rateB2F_prev

        self.transition_rateB2F = self.transition_rateB2F_prev
        self.transition_rateF2B = self.transition_rateF2B_prev

    def eval_act(self,s,a):
        logit = self.Q[s, :] 
        prob  = softmax(self.beta*logit)
        return prob[int(a)]

    def policy(self, s):
        logit = self.Q[s,:]
        return softmax(self.beta*logit)

    def beyesion_relest(self):
        self.MB_thr_PE = self.w*np.array([-1, 1]) # length = self.K-1

        ''' MB model reliability estitation'''  
        self.M_current_MB = np.min([self.M_current_MB+1, self
                                    .M]) # update # of accumulated events
        
        # (0) backup old values
        self.MB_mean_old = self.MB_mean
        self.MB_var_old = self.MB_var    
        self.MB_inv_Fano_old = self.MB_inv_Fano
        
        # (1) find the corresponding row
        PE_level = np.where((self.MB_thr_PE-self.agent_MB.SPE)<0); # [!!] must be fwd because it looks into SPE.    
        PE_theta = len(PE_level[0]); # 0:neg, 1:zero, 2:posPE
        
        # (2) update the current column(=1) in PE_history
        self.MB_PE_history[:,1:] = self.MB_PE_history[:,0:-1] # shift 1 column (toward past)
        self.MB_PE_history[:,0] = np.zeros(self.K) # empty the first column
        self.MB_PE_history[PE_theta,0] = 1 # add the count 1 in the first column
        self.MB_PE_num = np.sum(self.MB_PE_history == 1, axis=1)  # compute discounted accumulated PE
        
        # (3) posterior mean & var
        sumEvents = np.sum(self.MB_PE_num)
        sumEvents_excl = sumEvents-self.MB_PE_num
        self.MB_mean = (1+self.MB_PE_num)/(3+sumEvents)
        self.MB_var = ((1+self.MB_PE_num)*(2+sumEvents_excl))/((3+sumEvents)**2*(4+sumEvents)) 

        # (4) caculate reliability
        self.MB_triinv_Fano = self.MB_mean/self.MB_var
        self.MB_inv_Fano = self.MB_triinv_Fano[1]/sum(self.MB_triinv_Fano) 

        ''' MF model reliability estitation''' 
        # (0) backup old values

        self.MF_inv_Fano_old = self.MF_inv_Fano

        # (1) update of the absolute RPE estimator
        self.MF_absPEestimate = self.MF_absPEestimate+self.eta*(abs(self.agent_MF.RPE)-self.MF_absPEestimate)

        # (2) caculate reliability
        self.MF_inv_Fano = (40-self.MF_absPEestimate)/40 # [0,1]

    def Dynamic_Arbit(self): 
        input1 = self.MB_inv_Fano
        input2 = self.MF_inv_Fano
        
        self.transition_rateF2B = self.A_F2B/(1+np.exp(self.B_F2B*input2))
        self.transition_rateB2F = self.A_B2F/(1+np.exp(self.B_B2F*input1))

        self.transition_rateF2B_prev = self.transition_rateF2B
        self.transition_rateB2F_prev = self.transition_rateB2F


        self.tau = 1/(self.transition_rateF2B + self.transition_rateB2F) # alpha + beta term.
        self.MB_prob_prev = self.MB_prob
        self.MB_prob_inf = self.transition_rateF2B*self.tau
        self.MB_prob = self.MB_prob_inf+(self.MB_prob_prev-self.MB_prob_inf)*np.exp((-1)*self.time_step/self.tau)
        self.MF_prob = 1-self.MB_prob

        # choice of the model
        self.ind_active_model_prev = self.ind_active_model
        if self.MB_prob > 0.5:
            self.ind_active_model = 1
            self.num_MB_chosen = self.num_MB_chosen+1
            # there is no Q-value hand-over because sarsa computes Q based on SPE.
        else:
            self.ind_active_model = 2   
            self.num_MF_chosen = self.num_MF_chosen+1
            # Q-value hand-over : sarsa uses RPE-based Q only, does not use SPE.
        
    def learn(self): 
        self.agent_MF.mem = self.mem
        self.agent_MB.mem = self.mem
        _,_,self.Q_MF = self.agent_MF.learn() 
        _,_,self.Q_MB = self.agent_MB.learn() 
        self.beyesion_relest()
        self.Dynamic_Arbit()
        self.Q = ((self.MB_prob*self.Q_MB)**self.p + (self.MF_prob*self.Q_MF)**self.p)**(1/self.p)
    
    def bw_update(self,g): 
        self.Q_MB = self.agent_MB.bw_update(g)
        self.Q_MF = self.agent_MF.bw_update(g)
        self.Q = ((0.9*self.Q_MB)**self.p + (0.1*self.Q_MF)**self.p)**(1/self.p)


class Hybrid():
    name = 'Hybrid MF-MB'
    bnds = [(0,1),(0,5),(0,1)] # 边界
    pbnds = [(.1,.5),(.1,2),(.1,.9)] # 采样边界
    p_name   = ['alpha', 'beta', 'omega']  # 参数名
    n_params = len(p_name) 

    p_trans = [lambda x: 0.0 + (1 - 0.0) * sigmoid(x),   # alpha
               lambda x: 0.0 + (5 - 0.0) * sigmoid(x),   # beta
               lambda x: 0.0 + (1 - 0.0) * sigmoid(x)]   # omega (MF-MB weight)
    
    p_links = [lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_)),  
               lambda y: logit(np.clip((y - 0.0) / (5 - 0.0), eps_, 1 - eps_)),
               lambda y: logit(np.clip((y - 0.0) / (1 - 0.0), eps_, 1 - eps_))] 

    gamma = 1 
    
    def __init__(self, env, params):
        self.env = env 
        self.gamma = Hybrid.gamma
        self._init_mem()
        self._init_critic()
        self._load_params(params)
        self._load_agents()

    def _init_mem(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        params = [fn(p) for p, fn in zip(params, self.p_trans)]
        self.alpha = params[0]  # shared learning rate for both MF and MB
        self.beta  = params[1]  # inverse temperature 
        self.omega = params[2]  # weight between MF and MB (0=pure MF, 1=pure MB)

    def _load_agents(self):
        self.agent_MF = MF(self.env,params=[self.alpha,self.beta])
        self.agent_MB = MB(self.env,params=[self.alpha,self.beta])

    def _init_critic(self):
        self.Q = np.zeros([self.env.nS, self.env.nA])

    # ----------- decision ----------- #
    def policy(self, s):
        logit = self.Q[s, :]
        return softmax(self.beta*logit)

    def eval_act(self, s, a):
        '''Evaluate the probability of given state and action'''
        logit = self.Q[s, :] 
        prob  = softmax(self.beta*logit)
        return prob[int(a)]
    
    # ----------- learning ----------- #
    
    def learn(self):
        self.agent_MF.mem = self.mem
        self.agent_MB.mem = self.mem
        _,_,self.Q_MF = self.agent_MF.learn() 
        _,_,self.Q_MB = self.agent_MB.learn() 
        self.Q = self.omega*self.Q_MB + (1 - self.omega)*self.Q_MF

    def bw_update(self, g):
        self.Q_MB = self.agent_MB.bw_update(g)
        self.Q_MF = self.agent_MF.bw_update(g)
        self.Q = 0.9*self.Q_MB + 0.1*self.Q_MF
        return self.Q
