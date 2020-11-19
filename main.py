import farad as fd
import farad.elem as el
import farad.driver as ad
f = lambda x: el.exp(x) * el.sin(x**2)
function = ad.AutoDiff(f)
print(function.forward(1))