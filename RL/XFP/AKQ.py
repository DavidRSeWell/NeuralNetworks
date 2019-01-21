'''
    We are going ot apply the XFP algorithm to the AKQ game
    XFP: Extensive-form fictitious play algorithm

    Func:
        def FP(tree):
            init strategy
            j = 1
            while in budget do
                b(t +1) = ComputeBR(strategy)
                strategy(t + 1) = Update Avg Strat
                j += 1

            end
            return strategy
        end

        def ComputeBR(strategy)
            recurisve game tree
            return b(t + 1)
        end

        def UpdateAvgStrat(strategy(t), b(t+1))
            Theorem 7 from Heinrich paper
            or something simpler
           return strateg(t + 1)

        end

'''





def FP(tree):

    p1_init_strategy = {}

    p1_init_strategy = {}

    strategy_profile = []



