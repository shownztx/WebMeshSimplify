from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from djangoProject.simplify import main


def index(request):
    template = loader.get_template('../templates/test.html')
    context = {}
    return HttpResponse(template.render(context, request))


@csrf_exempt
def login(request):
    name = request.POST.get('name')
    print(name)
    return render(request, 'test.html')


@csrf_exempt
def simplify(request):
    mesh = request.POST.get('mesh')
    seg_factor = request.POST.get('seg_factor')
    simp_factor = request.POST.get('simp_factor')
    print(mesh, " - ", seg_factor, " - ", simp_factor)
    # main.main(str(mesh)+'.stl', seg_factor, simp_factor)
    return render(request, 'login.html')
