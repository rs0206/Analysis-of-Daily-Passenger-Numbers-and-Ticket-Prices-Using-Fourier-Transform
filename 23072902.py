#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:17:26 2024

@author: rahulyadav
"""

#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scrfft import scrfft  # Importing the custom scrfft module

# Loading the data
file_path = "airline0.csv"
data = pd.read_csv(file_path)

# Extracting year and month from Date column
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Fourier Transform using scrfft
xdata = data['Day']
ydata = data['Number']
frequencies, cosine_coeff, sine_coeff = scrfft(xdata, ydata)

# Calculating the Fourier series approximation for first 8 terms
def fourier_series_scrfft(x, f, a, b, num_terms=8):
    approx = np.zeros_like(x, dtype=np.float64)
    for k in range(num_terms):
        approx += a[k] * np.cos(2 * np.pi * f[k] * x) + b[k] * np.sin(2 * np.pi * f[k] * x)
    return approx

# Generating daily indices for one year (365 days)
days_in_year = 365
x_values = np.linspace(0, days_in_year-1, days_in_year)

# Mapping day indices to months for overlaying
x = x_values * (12 / days_in_year)  
x_round = np.round(x, 2)
print(f"After Mapping day indices to months for overlaying: {x_round}", end="\n\n")

# Fourier series approximation with 8 terms
fourier_approximation = fourier_series_scrfft(x_values, frequencies, cosine_coeff, sine_coeff, num_terms=8)

# Rounds the computed approximation values to 2 decimal places
fourier_approximation_round = np.round(fourier_approximation, 2)
print(fourier_approximation_round, end="\n\n")

# Average monthly passengers
monthly_avg_passengers = data.groupby('Month')['Number'].mean()

print(f"Monthly Average passengers: {monthly_avg_passengers}", end="\n\n")

# Figure 1: Average Monthly Passengers and Fourier Series Approximation
plt.figure(figsize=(14, 8))

# Bar Chart 
bars = plt.bar(
    x=monthly_avg_passengers.index,
    height=monthly_avg_passengers,
    color='gold',  
    edgecolor='black',
    label='Average Monthly Passengers',
    tick_label=[
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
)

# Plot of Fourier series approximation
plt.plot(
    x_values * (12 / days_in_year),  # Mapping day indices to months for overlaying
    fourier_approximation,
    color='red',
    linewidth=2,
    label='Fourier Series Approximation (8 terms)'
)

# Adding the student ID to the top-left corner of the plot
plt.text( 0.05, 0.95, 'Student ID: 23072902', 
         fontsize=14, color='darkred', 
         ha='left', va='top', transform=plt.gca().transAxes
)

# Customizing the plot
plt.title('Average Daily Passengers and Fourier Series Approximation', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Average Number of Passengers', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=14)

# Display the plot for Figure 1
plt.tight_layout()
plt.show()


# Figure 2: Average Ticket Price and Power Spectrum
# Calculating the power spectrum from the Fourier coefficients
power_spectrum = cosine_coeff**2 + sine_coeff**2

# Identifying the main period (Y) corresponding to the highest power
main_period_idx = np.argmax(power_spectrum[1:]) + 1  # Skip the zero frequency component
main_period_frequency = frequencies[main_period_idx]
Y = 1 / main_period_frequency  # Convert frequency to period (days)

# Calculate the average ticket price in 2022 (X)
data_2022 = data[data['Year'] == 2022]  # Filter for 2022 data
X = data_2022['Price'].mean()  # Average ticket price for 2022

# Plotting the power spectrum (Figure 2)
plt.figure(figsize=(14, 8))
plt.plot(
    frequencies[1:],  # Exclude the zero frequency
    power_spectrum[1:],  # Exclude the zero frequency
    color='purple',
    alpha=0.7,
    label='Power Spectrum'
)

# Highlighting the main period
plt.axvline(main_period_frequency, color='red', linestyle='--', label=f'Main Period ({Y:.2f} days)')

# Displaying X (average ticket price) and Y (main period) on the plot
plt.text(
    0.02, 0.95,
    f'Average Ticket Price (X): ${X:.2f}\nMain Period (Y): {Y:.2f} days',
    transform=plt.gca().transAxes,
    fontsize=14,
    color='darkblue',
    verticalalignment='top'
)

# Adding student ID to the top-left corner of the plot
plt.text(
    0.02, 0.85,
    'Student ID: 23072902',
    transform=plt.gca().transAxes,
    fontsize=14,
    color='darkgreen',
    verticalalignment='top'
)

# Customizing the plot
plt.title('Power Spectrum of Daily Passenger Numbers', fontsize=16)
plt.xlabel('Frequency (1/day)', fontsize=14)
plt.ylabel('Power', fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot for Figure 2
plt.show()

# Output the values for X and Y
X, Y
