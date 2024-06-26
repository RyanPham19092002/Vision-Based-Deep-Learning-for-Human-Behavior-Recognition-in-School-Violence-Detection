from django.shortcuts import render, redirect
# from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages


import datetime

from .forms import CreateUserForm

# Create your views here.

def loginPage(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('cameras')
        else:
            messages.info(request, "Username or password is incorrect")

    context = {}
    return render(request, 'accounts/login.html', context)

def registerPage(request):
    if request.user.is_authenticated:
        return redirect('cameras')
    else:
        form = CreateUserForm()

        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, "Account was created for " + user)

                return redirect('login')

    context = {"form": form}
    return render(request, "accounts/register.html", context)

def logoutUser(request):
    logout(request)
    return redirect('login')




