module NN
using Knet

function predict(w,x)
    for i=1:2:length(w)
        #println(size(w[i]),size(w[i+1]))
        x = w[i]*mat(x) .+ w[i+1]
        if i<length(w)-1
            x = relu.(x)
        end
    end
    #x = sum(x,1)
    #println(size(x))
    return x
end

function loss(w,x,y)
    #println(size(y))
    ypred = predict(w,x)
    l = sum((y-ypred).^2)/length(y)
    return l
end

    
lossgradient = grad(loss)

function train(w, dtrn, xtrain, ytrain, mu=1e-2, epochs=10)
    p = optimizers(w,Nesterov; lr = mu)
    for epoch=1:epochs
        
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
            update!(w,g,p)
        end
        if epoch%1000 == 0
            report(epoch,w,xtrain,ytrain)
        end
    end
    return w
end

function init_weight(hidden_size)
    w = Any[]
    for h in hidden_size
        push!(w, randn(h[2],h[1]))
        push!(w, randn(h[2],1))
    end
    
    return w
end

report(epoch,w,x,y)=println((:epoch,epoch,:trn,loss(w,x,y)))

function generate_data(a,r,n)
    theta = 4*pi*rand(n,)
    x = zeros(n,2)
    for i = 1:n
        x[i,1] = a*(theta[i]^(1/r))*sin(theta[i])
        x[i,2] = a*(theta[i]^(1/r))*cos(theta[i])
    end
    
    return x
end

function generate_data(n,m)
    x1 = generate_data(1,1,n) + randn(n,2)/2
    x2 = generate_data(-1,1,n) + randn(n,2)/2
    
    xtrain = vcat(x1,x2)'
    ytrain = vcat(ones(n,1),zeros(n,1))'
    
    dtrn = minibatch(xtrain,ytrain,m;shuffle=true)
    
    return dtrn,xtrain,ytrain
end

function boundary(w)
    predict_plane = [ sign.(predict(w,mat([[y1];[y2]]))[1,1]-0.5)  for y1 = linspace(-10,10,50), y2 = linspace(-10,10,50)]
    return predict_plane
end

end