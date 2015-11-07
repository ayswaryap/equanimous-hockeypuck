

clc;
clear all;hold all;

rng(0);
nTransmit = 8;
nReceive = 1;

SNR_dB = 10;
nBases = 2;
nUsers = 4;
sPower = 10^(SNR_dB/10);

maxIterate = 50;
epsilonThr = 1e-3;
userBase = zeros(nUsers,1);
cellUserIndices = {[1 2]' ; [3 4]'};
cH = complex(randn(nReceive,nTransmit,nUsers,nBases),randn(nReceive,nTransmit,nUsers,nBases)) / sqrt(2);

for iUser = 1:nUsers
    for iBase = 1:nBases
        if sum(iUser == cellUserIndices{iBase,1})
            userBase(iUser,1) = iBase;
            break;
        end
    end
end

cvx_quiet('true');
cvx_solver('Sedumi');
cvx_expert('true');

%% Centralized Problem

cIterate = 0;
reIterate = 1;
prevRate = 1e5;
phi = rand(nUsers,1);
sumRateIterate = zeros(maxIterate,1);

while reIterate
    
    cvx_begin
    
    variable M(nTransmit,nUsers) complex
    variables t(nUsers,1) p(nUsers,1) q(nUsers,1) b(nUsers,1) g(nUsers,1)
    
    maximize(sum(t))
    
    subject to
    
    for iUser = 1:nUsers
        log2(exp(1)) * log(1 + g(iUser,1)) >= t(iUser,1);
    end
    
    for iUser = 1:nUsers
        
        intVector = 1;
        imag(cH(:,:,iUser,iBase) * M(:,iUser)) == 0;
        2 * real(cH(:,:,iUser,userBase(iUser,1)) * M(:,iUser)) >= (1/phi(iUser,1)) * g(iUser,1) + phi(iUser,1) * b(iUser,1)^2;
        for jUser = 1:nUsers
            if jUser ~= iUser
                intVector = [intVector ; cH(:,:,iUser,userBase(jUser,1)) * M(:,jUser)];
            end
        end
        
        norm(intVector,2) <= b(iUser,1);
        for iBase = 1:nBases
            norm(vec(M(:,cellUserIndices{iBase,1}))) <= sPower;
        end
        
    end
    
    cvx_end
    
    display(sum(t));
    phi = sqrt(g) ./ b;
    
    cIterate = cIterate + 1;
    if abs(prevRate - sum(t)) < epsilonThr
        reIterate = 0;
    else
        prevRate = sum(t);
    end
    
    sumRateIterate(cIterate,1) = sum(t);
    if cIterate >= maxIterate
        reIterate = 0;
    end
    
end

plot(sumRateIterate(1:cIterate),'b');
keyboard;

%% Distributed ADMM

rho = 2;
cIterate = 0;
reIterate = 1;
prevRate = 1e5;
phi = rand(nUsers,1);
sumRateIterate = zeros(maxIterate,1);

maxADMMIterations = 5;
globalIFThreshold = zeros(nUsers,nBases);
cB = cell(nBases,1);cG = cell(nBases,1);
cDual = cell(nBases,1);cellIF = cell(nBases,1);cellT = cell(nBases,1);

for iBase = 1:nBases
    cDual{iBase,1} = zeros(nUsers,nBases);
end

while reIterate
    
    admmIterate = 1;
    cADMMIterate = 0;
        
    while admmIterate
    
        for iBase = 1:nBases

            kUsers = length(cellUserIndices{iBase,1});

            cvx_begin

            variable M(nTransmit,kUsers) complex
            variables t(kUsers,1) p(kUsers,1) q(kUsers,1) b(kUsers,1) g(kUsers,1) ifThreshold(nUsers,nBases) epiG

            maximize(epiG);

            subject to

            for iUser = 1:kUsers
                log2(exp(1)) * log(1 + g(iUser,1)) >= t(iUser,1);
            end

            epiG <= sum(t) + sum(vec(cDual{iBase,1}.*ifThreshold)) - (rho/2)*sum(vec(ifThreshold - globalIFThreshold).^2);

            for iUser = 1:kUsers

                cUser = cellUserIndices{iBase,1}(iUser,1);
                2 * real(cH(:,:,cUser,iBase) * M(:,iUser)) >= (1/phi(cUser,1)) * g(iUser,1) + phi(cUser,1) * b(iUser,1)^2;
                imag(cH(:,:,cUser,iBase) * M(:,iUser)) == 0;
                
                intVector = 1;
                for jUser = 1:kUsers
                    if jUser ~= iUser
                        intVector = [intVector ; cH(:,:,cUser,iBase) * M(:,jUser)];
                    end
                end

                for jBase = 1:nBases
                    if iBase ~= jBase
                        intVector = [intVector ; ifThreshold(cUser,jBase)];
                    end
                end
                
                norm(intVector,2) <= b(iUser,1);

            end

            for jBase = 1:nBases
                if jBase ~= iBase
                    for jUser = 1:length(cellUserIndices{jBase,1})
                        ifJuser = cellUserIndices{jBase,1}(jUser,1);
                        norm(cH(:,:,ifJuser,iBase) * M,2) <= ifThreshold(ifJuser,iBase);
                    end
                end            
            end            

            norm(vec(M(:))) <= sPower;


            cvx_end
    
            cB{iBase,1} = b;cG{iBase,1} = g;
            cellIF{iBase,1} = ifThreshold;
            cellT{iBase,1} = t;        

        end
        
        for iBase = 1:nBases
            for iUser = 1:length(cellUserIndices{iBase,1})
                cUser = cellUserIndices{iBase,1}(iUser,1);
                for jBase = 1:nBases
                    if jBase ~= iBase
                        globalIFThreshold(cUser,jBase) = (cellIF{iBase,1}(cUser,jBase) + cellIF{jBase,1}(cUser,jBase)) * 0.5;
                    end
                end
            end
        end
        
        for iBase = 1:nBases
            for iUser = 1:length(cellUserIndices{iBase,1})
                cUser = cellUserIndices{iBase,1}(iUser,1);
                for jBase = 1:nBases
                    if jBase ~= iBase
                        cDual{iBase,1}(cUser,jBase) = cDual{iBase,1}(cUser,jBase) - rho * (cellIF{iBase,1}(cUser,jBase) - globalIFThreshold(cUser,jBase));
                    end
                end
            end
            
            for iUser = 1:nUsers
                if ~sum(iUser == cellUserIndices{iBase,1})
                    cDual{iBase,1}(iUser,iBase) = cDual{iBase,1}(iUser,iBase) - rho * (cellIF{iBase,1}(iUser,iBase) - globalIFThreshold(iUser,iBase));
                end
            end
            
        end
        
        display(sum(cell2mat(cellT)));
        
        cADMMIterate = cADMMIterate + 1;
        if cADMMIterate >= maxADMMIterations
            admmIterate = 0;
        end
        
    end
    
    for iBase = 1:nBases
        phi(cellUserIndices{iBase,1},1) = sqrt(cG{iBase,1}) ./ cB{iBase,1};
    end
    
    cIterate = cIterate + 1;
    if abs(prevRate - sum(t)) < epsilonThr
        reIterate = 0;
    else
        prevRate = sum(cell2mat(cellT));
    end
    
    sumRateIterate(cIterate,1) = sum(cell2mat(cellT));
    if cIterate >= maxIterate
        reIterate = 0;
    end
    
end

plot(sumRateIterate(1:cIterate),'r');
keyboard;

%%


