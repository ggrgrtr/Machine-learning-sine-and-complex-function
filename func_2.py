import torch
import matplotlib.pyplot as plt


def func_2(x):
    return 2**x*torch.sin(2**(-x))

x_test=torch.rand(100)
y_test=func_2(x_test)

plt.title('tests')
plt.xlabel('$x_t$')
plt.ylabel('$y_t$')
plt.plot(x_test.numpy(),y_test.numpy(),'r-')
plt.plot(x_test.numpy(),y_test.numpy(),'b-')
plt.show()


x_test.unsqueeze_(1)
y_test.unsqueeze_(1)

x_valid=torch.linspace(-10,10,100)
y_valid=func_2(x_valid)
x_valid.unsqueeze_(1)
y_valid.unsqueeze_(1)


def loss_f(pred,t):
    return torch.mean((pred-t)**2)

class func_net(torch.nn.Module):
    def __init__(self,n):
        super(func_net, self).__init__()
        self.neurons1=torch.nn.Linear(1,n)
        self.act_f=torch.nn.Sigmoid()
        self.neurons2=torch.nn.Linear(n,1)

    def forward(self,x):
        x=self.neurons1(x)
        x=self.act_f(x)
        x=self.neurons2(x)
        return x

def prediction(net,x,y):
    y_pr=net.forward(x)
    plt.plot(x,y,'r-')
    plt.plot(x.numpy(),y_pr.data,'b-')
    plt.axis([-11, 11, -1.1, 1.1])
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()


my_net=func_net(10)
grad_step=torch.optim.Adam(my_net.parameters(),lr=0.01)

for epoch in range(100):
    grad_step.zero_grad()
    y_pred=my_net.forward(x_test)
    loss=loss_f(y_pred,y_test)
    loss.backward()
    grad_step.step()

prediction(my_net,x_valid,y_valid)
