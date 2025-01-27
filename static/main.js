// Configuration for API base URL
const baseUrl = window.location.origin;

// Check if jQuery is loaded
if (typeof $ === 'undefined') {
    console.error('jQuery is not loaded. Skipping script execution.');
    alert("The application requires jQuery to function properly.");
} else {
    console.log('jQuery loaded successfully.');

    $(document).ready(function () {
        const currentPath = window.location.pathname;
        console.log("Current Path:", currentPath);
        
        $('.sidebar .nav-item').each(function () {
            const link = $(this).find('a.nav-link').attr('href');
            if (link === currentPath) {
                $(this).addClass('active');
            } else {
                $(this).removeClass('active');
            }
        });

        if (currentPath === '/mlops_predictionevaluation') {
           
            initializeMetricsOverviewChart();
            initializePredictionTrendsChart();
            initializeMetricsSummaryTable();
            initializePredictionComparisonTable();
            fetchAndUpdateBestModelUI();
            initializeModelComparisonChart();
            initializeRadarChart();
            
        }

        if (currentPath === '/mlops_preprocessing') {
            initializeDatasetPreviewRefresh();   
            initializeDatasetPreviewTable();
        }
        
        if (currentPath === '/mlops_trainingvalidation') {
            initializeTrainingLogs();
            initializeValidationLogs();
            initializeRefreshButtons();
        }
        
       
        if (currentPath === '/eda/holiday') {
            console.log('Initializing Holiday EDA...');
            fetchAndRenderHolidayData(); // Fetch and render EDA Holiday data
            
        }
    
        if (currentPath === '/eda/weather') {
            console.log('Initializing Weather EDA...');
            fetchAndRenderWeatherData(); // Fetch and render EDA Weather data
        }

        if (currentPath === '/eda/demand') {
            console.log('Initializing Demand EDA...');
            fetchAndRenderDemandData();
        }
    
        if (currentPath === '/hourlydemand') {
            console.log('Initializing Hourly Demand Table...');
            initializeHourlyDemandTable(); // Initialize Hourly Demand Table
        }
    
        if (currentPath === '/hourlyweather') {
            console.log('Initializing Hourly Weather Table...');
            initializeHourlyWeatherTable(); // Initialize Hourly Weather Table
        }
    
        if (currentPath === '/holidays') {
            console.log('Initializing Holidays Page...');
            initializeHolidaysTable(); // Initialize Holidays Table
        }
    
      
        // Common initialization
        console.log('Initializing Global Functions...');
        initializeAutoUpdate(); // Auto-update the "Last Updated" section
        initializeSearchInput(); // Search functionality for input fields
        initializeMenuSearch(); // Sidebar menu search functionality
    });
    
}

// 1. START SECTION 1 MENU AND SEARCH

// Initialize Sidebar Menu Search
function initializeMenuSearch() {
    const menuSearchId = '#menuSearch';
    if ($(menuSearchId).length) {
        $(menuSearchId).on('input', function () {
            const query = $(this).val().toLowerCase().trim();
            $('.sidebar .nav-item').each(function () {
                const text = $(this).text().toLowerCase();
                $(this).toggle(text.includes(query));
            });
        });
    }
}


// Initialize Global Search Input for Tables
function initializeSearchInput() {
    const searchInputId = '#searchInput'; // Global search input element

    if ($(searchInputId).length) {
        let debounceTimer;

        $(searchInputId).on('input', function () {
            clearTimeout(debounceTimer); // Clear previous debounce timer
            const searchValue = $(this).val().trim();

            debounceTimer = setTimeout(() => {
                // Apply search to DataTables if available
                if ($('#hourly-demand-table').length) {
                    $('#hourly-demand-table').DataTable().search(searchValue).draw();
                }
                if ($('#hourly-weather-table').length) {
                    $('#hourly-weather-table').DataTable().search(searchValue).draw();
                }
                if ($('#holidays-table').length) {
                    $('#holidays-table').DataTable().search(searchValue).draw();
                }
            }, 300); // Debounce: Wait for 300ms after user stops typing
        });
    }
}

// 1. END SECTION 1 MENU AND SEARCH

// 2. START SECTION 2 DASHBOARD

// Fetch and Render Dashboard Data

/*
// Render Dashboard Data
function renderDashboardData(data) {
    // Assuming widgets for demand and weather summary exist in the DOM
    const totalDemandWidgetId = '#total-demand-widget';
    const averageTemperatureWidgetId = '#average-temperature-widget';

    const totalDemand = data.reduce((acc, item) => acc + item.total_demand, 0);
    const averageTemperature =
        data.length > 0
            ? data.reduce((acc, item) => acc + item.average_temperature, 0) / data.length
            : 0;

    $(totalDemandWidgetId).text(`Total Demand: ${totalDemand.toFixed(2)} kWh`);
    $(averageTemperatureWidgetId).text(`Avg Temperature: ${averageTemperature.toFixed(2)} °C`);
}

// Initialize Dashboard
function initializeDashboard() {
    fetchAndRenderDashboard();
}*/

// 2. END SECTION 2 DASHBOARD

// 3. START SECTION 3 HOURLY DEMAND

// Initialize Hourly Demand Table
function initializeHourlyDemandTable() {
    const tableId = '#hourly-demand-table';

    if ($(tableId).length) {
        // Destroy existing DataTable if initialized
        if ($.fn.DataTable.isDataTable(tableId)) {
            $(tableId).DataTable().destroy();
        }

        // Initialize new DataTable
        $(tableId).DataTable({
            processing: true, // Show loading spinner
            serverSide: true, // Fetch data server-side
            ajax: {
                url: `${baseUrl}/api/hourlydemand`, // API endpoint
                type: 'GET',
                dataSrc: function (json) {
                    if (json && json.data) {
                        return json.data; // Valid response
                    } else {
                        console.error("Invalid Hourly Demand JSON response:", json);
                        return []; // Fallback to empty array
                    }
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error('Error fetching Hourly Demand data:', {
                        status: jqXHR.status,
                        responseText: jqXHR.responseText,
                        textStatus,
                        errorThrown,
                    });
                    alert("Failed to load Hourly Demand data. Please try again.");
                },
            },
            columns: [
                { data: 'time', title: 'Datetime' }, // Time column
                { data: 'value', title: 'Demand (kWh)' }, // Demand value column
            ],
            order: [[0, 'desc']], // Default order by datetime descending
            pageLength: 10, // Number of rows per page
        });
    }
}
/*
// Render Hourly Demand Page
function renderHourlyDemandPage() {
    fetch(`${baseUrl}/hourlydemand`)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.text();
        })
        .then((html) => {
            document.querySelector("#content-wrapper").innerHTML = html;

            // Reinitialize DataTable after content is loaded
            initializeHourlyDemandTable();
        })
        .catch((error) => {
            console.error("Error rendering Hourly Demand page:", error);
            alert("Failed to load Hourly Demand page. Please try again later.");
        });
}

// Event Listener for Hourly Demand Navigation
document.addEventListener("DOMContentLoaded", function () {
    const hourlyDemandNav = document.querySelector("a[href='/hourlydemand']");

    if (hourlyDemandNav) {
        hourlyDemandNav.addEventListener("click", function (e) {
            e.preventDefault();
            renderHourlyDemandPage();
        });
    }
});*/

// 3. END SECTION 3 HOURLY DEMAND

// 4. START SECTION 4 HOURLY WEATHER

// Initialize Hourly Weather Table
function initializeHourlyWeatherTable() {
    const tableId = '#hourly-weather-table';

    if ($(tableId).length) {
        // Destroy existing DataTable if already initialized
        if ($.fn.DataTable.isDataTable(tableId)) {
            $(tableId).DataTable().destroy();
        }

        // Initialize new DataTable
        $(tableId).DataTable({
            processing: true, // Show loading spinner
            serverSide: true, // Fetch data server-side
            ajax: {
                url: `${baseUrl}/api/hourlyweather`, // API endpoint
                type: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                dataSrc: function (json) {
                    if (json && json.data) {
                        return json.data; // Valid response
                    } else {
                        console.error("Invalid Hourly Weather JSON response:", json);
                        return []; // Fallback to empty array
                    }
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error('Error fetching Hourly Weather data:', {
                        status: jqXHR.status,
                        responseText: jqXHR.responseText,
                        textStatus,
                        errorThrown,
                    });
                    alert("Failed to load Hourly Weather data. Please try again.");
                },
            },
            columns: [
                { data: 'datetime', title: 'Datetime' }, // Datetime column
                { data: 'temp', title: 'Temperature (°C)' }, // Temperature column
                { data: 'feelslike', title: 'Feels Like (°C)' }, // Feels like column
                { data: 'humidity', title: 'Humidity (%)' }, // Humidity column
                { data: 'windspeed', title: 'Wind Speed (km/h)' }, // Wind speed column
                { data: 'cloudcover', title: 'Cloud Cover (%)' }, // Cloud cover column
                { data: 'solaradiation', title: 'Solar Radiation (W/m²)' }, // Solar radiation column
                { data: 'precip', title: 'Precipitation (mm)' }, // Precipitation column
                { data: 'preciptype', title: 'Precipitation Type' }, // Precipitation type column
            ],
            order: [[0, 'desc']], // Default order by datetime descending
            pageLength: 10, // Number of rows per page
        });
    }
}
/*
// Render Hourly Weather Page
function renderHourlyWeatherPage() {
    fetch(`${baseUrl}/hourlyweather`)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.text();
        })
        .then((html) => {
            document.querySelector("#content-wrapper").innerHTML = html;

            // Reinitialize DataTable after content is loaded
            initializeHourlyWeatherTable();
        })
        .catch((error) => {
            console.error("Error rendering Hourly Weather page:", error);
            alert("Failed to load Hourly Weather page. Please try again later.");
        });
}

// Event Listener for Hourly Weather Navigation
document.addEventListener("DOMContentLoaded", function () {
    const hourlyWeatherNav = document.querySelector("a[href='/hourlyweather']");

    if (hourlyWeatherNav) {
        hourlyWeatherNav.addEventListener("click", function (e) {
            e.preventDefault();
            renderHourlyWeatherPage();
        });
    }
});
*/
// 4. END SECTION 4 HOURLY WEATHER

// 5. START SECTION 5 HOLIDAYS

// Initialize Holidays Table
function initializeHolidaysTable() {
    const tableId = '#holidays-table';

    if ($(tableId).length) {
        // Destroy existing DataTable if already initialized
        if ($.fn.DataTable.isDataTable(tableId)) {
            $(tableId).DataTable().destroy();
        }

        // Initialize new DataTable
        $(tableId).DataTable({
            serverSide: true, // Fetch data server-side
            processing: true, // Show loading spinner
            ajax: {
                url: `${baseUrl}/api/holidays`, // API endpoint
                type: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                dataSrc: function (json) {
                    if (json && json.data) {
                        return json.data; // Valid response
                    } else {
                        console.error("Invalid Holidays JSON response:", json);
                        return []; // Fallback to empty array
                    }
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error('Error fetching Holidays data:', {
                        status: jqXHR.status,
                        responseText: jqXHR.responseText,
                        textStatus,
                        errorThrown,
                    });
                    alert("Failed to load Holidays data. Please try again.");
                },
            },
            columns: [
                { data: 'date', title: 'Date' }, // Date column
                { data: 'name', title: 'Holiday Name' }, // Holiday name column
            ],
            order: [[0, 'desc']], // Default order by date descending
            pageLength: 10, // Number of rows per page
        });
    }
}

/*
// Render Holidays Page
function renderHolidaysPage() {
    fetch(`${baseUrl}/holidays`)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.text();
        })
        .then((html) => {
            document.querySelector("#content-wrapper").innerHTML = html;

            // Reinitialize DataTable after content is loaded
            initializeHolidaysTable();
        })
        .catch((error) => {
            console.error("Error rendering Holidays page:", error);
            alert("Failed to load Holidays page. Please try again later.");
        });
}

// Event Listener for Holidays Navigation
document.addEventListener("DOMContentLoaded", function () {
    const holidaysNav = document.querySelector("a[href='/holidays']");

    if (holidaysNav) {
        holidaysNav.addEventListener("click", function (e) {
            e.preventDefault();
            renderHolidaysPage();
        });
    }
});*/

// 5. END SECTION 5 HOLIDAYS

// 6. START SECTION 6 LAST UPDATED

// Update Last Updated Section
function updateLastUpdated() {
    // Fetch last updated timestamps from the API
    $.ajax({
        url: `${baseUrl}/api/lastUpdated`, // API endpoint
        type: 'GET',
        success: ({ lastUpdatedDemand, lastUpdatedWeather, lastUpdatedHoliday }) => {
            // Dynamically update the DOM elements
            $('#last-updated-demand em').text(
                lastUpdatedDemand ? `Last updated on ${lastUpdatedDemand}` : 'No updates available.'
            );
            $('#last-updated-weather em').text(
                lastUpdatedWeather ? `Last updated on ${lastUpdatedWeather}` : 'No updates available.'
            );
            $('#last-updated-holiday em').text(
                lastUpdatedHoliday ? `Last updated on ${lastUpdatedHoliday}` : 'No updates available.'
            );

            console.info("Last Updated section successfully refreshed.");
        },
        error: ({ status, responseText, statusText }) => {
            console.error('Error fetching Last Updated data:', {
                status,
                responseText,
                statusText,
            });
            alert("Unable to fetch 'Last Updated' data. Please try again later.");
        },
    });
}

// Initialize Auto Update for Last Updated Section
function initializeAutoUpdate() {
    updateLastUpdated(); // Initial call to fetch timestamps
    setInterval(updateLastUpdated, 60000); // Refresh every 60 seconds
}

// 6. END SECTION 6 LAST UPDATED

// 7. START SECTION 7 INITIALIZE TABLES
function initializeTables() {
    let initialized = false;

    if ($('#hourly-demand-table').length) {
        console.log("Initializing Hourly Demand Table...");
        initializeHourlyDemandTable();
        initialized = true;
    }

    if ($('#hourly-weather-table').length) {
        console.log("Initializing Hourly Weather Table...");
        initializeHourlyWeatherTable();
        initialized = true;
    }

    if ($('#holidays-table').length) {
        console.log("Initializing Holidays Table...");
        initializeHolidaysTable();
        initialized = true;
    }

    if (!initialized) {
        console.warn("No tables found to initialize. Please check the DOM or table IDs.");
    }
}

// 7. END SECTION 7 INITIALIZE TABLES

// 8. START SECTION 8 EDA HOLIDAYS

function fetchAndRenderHolidayData() {
    $.ajax({
        url:  `${baseUrl}/eda/holiday`,
        type: 'GET',
        dataType: 'json',
        success: function (response) {
            // Update HTML elements
            $('#total-holidays').text(response.total_holidays || 'N/A');
            $('#common-month').text(response.common_month || 'N/A');

            // 1. Line Chart: Holiday Trends Over the Years
            renderPlotlyChart('holiday-trends', response.holiday_trends, {
                xaxis: { title: 'Year', showgrid: true, zeroline: false },
                yaxis: { title: 'Total Holidays', showgrid: true, zeroline: false }
            }, (data) => ({
                x: data.map(d => d.year),
                y: data.map(d => d.total_holidays),
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'spline', color: '#007bff', width: 3 },
                marker: { size: 8, color: '#007bff' },
            }));

            // 2. Treemap: Holiday Count by Day of the Week
            renderPlotlyChart('holidays-by-day', response.holidays_by_day, {}, (data) => ({
                labels: data.map(d => d.day),
                parents: Array(data.length).fill(''),
                values: data.map(d => d.count),
                type: 'treemap',
            }));

            // 3. Stacked Bar Chart: Top Holidays Per Year
            const years = [...new Set(response.top_holidays_per_year.map(d => d.year))];
            const traces = years.map(year => {
                const filteredData = response.top_holidays_per_year.filter(d => d.year === year);
                return {
                    x: filteredData.map(d => d.name),
                    y: filteredData.map(d => d.count),
                    name: `Year ${year}`,
                    type: 'bar',
                };
            });
            Plotly.newPlot('top-holidays', traces, {
                barmode: 'stack',
                yaxis: { title: 'Count' },
            });

            // 4. Heatmap: Holiday Frequency
            renderPlotlyChart('holiday-heatmap', response.heatmap_data, {
                xaxis: { title: 'Month' },
                yaxis: { title: 'Year' },
            }, (data) => ({
                z: data,
                x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y: years.map(String),
                type: 'heatmap',
            }));

            // 5. Pie Chart: Holiday Distribution by Month
            renderPlotlyChart('holiday-pie-chart', response.monthly_distribution, {}, (data) => ({
                labels: data.map(d => getMonthName(d.month)),
                values: data.map(d => d.percentage),
                type: 'pie',
            }));
        },
        error: function (error) {
            console.error("Failed to fetch holiday data:", error);
        },
    });
}

function renderPlotlyChart(containerId, data, layout, processData) {
    const chartData = processData(data);
    Plotly.newPlot(containerId, Array.isArray(chartData) ? chartData : [chartData], layout);
}

function getMonthName(monthNumber) {
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    return monthNames[monthNumber - 1];
}




// 8. END SECTION 8 EDA HOLIDAYS



// 9. START SECTION 9 EDA WEATHER
function fetchAndRenderWeatherData() {
    $.ajax({
        url: `${baseUrl}/eda/weather`,
        type: 'GET',
        dataType: 'json',
        success: function (response) {
            // Update Overview Cards
            $('#highest-temp').text(`${response.highest_temp || 'N/A'} °C`);
            $('#lowest-temp').text(`${response.lowest_temp || 'N/A'} °C`);
            $('#highest-wind-speed').text(`${response.highest_wind_speed || 'N/A'} m/s`);
            $('#total-precipitation').text(`${response.total_precipitation || 'N/A'} mm`);
            $('#avg-solar-radiation').text(`${response.avg_solar_radiation.toFixed(2) || 'N/A'} W/m²`);
            $('#most-frequent-precip').text(response.most_frequent_precip_type || 'N/A');


            // 1. Line Chart: Average Temperature by Year
            renderPlotlyChart('avg-temp-by-year', response.avg_temp_by_year, {
                xaxis: { title: 'Year', showgrid: true, zeroline: false },
                yaxis: { title: 'Temperature (°C)', showgrid: true, zeroline: false },
            }, (data) => ({
                x: data.map(d => d.year),
                y: data.map(d => d.avg_temp),
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'spline', color: '#007bff', width: 3 },
                marker: { size: 8, color: '#007bff' },
            }));

            // 2. Bar Chart: Average Humidity by Month
            renderPlotlyChart('avg-humidity-by-month', response.avg_humidity_by_month, {
                xaxis: { title: 'Month', showgrid: true, zeroline: false },
                yaxis: { title: 'Humidity (%)', showgrid: true, zeroline: false },
            }, (data) => ({
                x: data.map(d => d.month),
                y: data.map(d => d.avg_humidity),
                type: 'bar',
                marker: { color: '#66BB6A' },
            }));

            // 3. Bar Chart: Wind Speed Distribution
            renderPlotlyChart('wind-speed-distribution', response.wind_speed_distribution, {
                xaxis: { title: 'Wind Speed (m/s)', showgrid: true, zeroline: false },
                yaxis: { title: 'Frequency', showgrid: true, zeroline: false },
            }, (data) => ({
                x: data.map(d => d.windspeed),
                y: data.map(d => d.count),
                type: 'bar',
                marker: { color: '#FF7043' },
            }));

            // 4. Line Chart: Solar Radiation by Hour
            renderPlotlyChart('solar-radiation-by-hour', response.solar_radiation_by_hour, {
                xaxis: { title: 'Hour of Day', showgrid: true, zeroline: false },
                yaxis: { title: 'Solar Radiation (W/m²)', showgrid: true, zeroline: false },
            }, (data) => ({
                x: data.map(d => d.hour),
                y: data.map(d => d.avg_radiation),
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'spline', color: '#FFA726', width: 3 },
                marker: { size: 8, color: '#FFA726' },
            }));

            // 5. Pie Chart: Precipitation Type Distribution
            renderPlotlyChart('precip-type-distribution', response.precip_type_distribution, {
            }, (data) => ({
                labels: data.map(d => d.type),
                values: data.map(d => d.percentage),
                type: 'pie',
            }));

            // New Insights
            // 6. Heatmap: Monthly Average Temperature (Seasonal Trends)
            renderPlotlyChart('monthly-avg-temp', response.monthly_avg_temp, {
                xaxis: { title: 'Month', showgrid: true },
                yaxis: { title: 'Year', showgrid: true },
                coloraxis: { colorscale: 'Blues' },
            }, (data) => ({
                z: data.map(d => d.temp),
                x: data.map(d => d.month),
                y: data.map(d => d.year),
                type: 'heatmap',
            }));

            // 7. Box Plot: Daily Humidity Distribution
            renderPlotlyChart('daily-avg-humidity', response.daily_avg_humidity, {
               
                yaxis: { title: 'Humidity (%)', showgrid: true, zeroline: false },
            }, (data) => ({
                x: data.map(d => d.day),
                y: data.map(d => d.avg_humidity),
                type: 'box',
                boxpoints: 'all',
            }));

            // 8. Scatter Plot: Hourly Wind Speed Variations
            renderPlotlyChart('hourly-avg-windspeed', response.hourly_avg_windspeed, {
                xaxis: { title: 'Hour', showgrid: true, zeroline: false },
                yaxis: { title: 'Wind Speed (m/s)', showgrid: true, zeroline: false },
            }, (data) => ({
                x: data.map(d => d.hour),
                y: data.map(d => d.avg_windspeed),
                mode: 'markers',
                type: 'scatter',
                marker: { size: 8, color: '#FF5722' },
            }));
        },
        error: function (error) {
            console.error("Failed to fetch weather data:", error);
        },
    });
}

// 9. END SECTION 9 EDA WEATHER

// 10. START SECTION 10 EDA DEMAND

function fetchAndRenderDemandData() {
    // Show a loading indicator while fetching data
    $('#loading-spinner').show();

    $.ajax({
        url: `${baseUrl}/eda/demand`, // Backend API endpoint
        method: 'GET',               // HTTP method
        dataType: 'json',            // Expected data format
        success: function (response) {
            console.log("Demand data fetched successfully:", response);

            try {
                // Validate response structure
                if (!response || !response.total_demand_per_year || !response.avg_daily_demand) {
                    throw new Error('Invalid response format');
                }

                // Hide error messages
                $('#error-message').hide();

                // Update Overview Cards
                $('#total-demand').text(response.total_demand_per_year.reduce((sum, d) => sum + d.total_demand, 0).toFixed(2));
                $('#average-daily-demand').text(response.avg_daily_demand.toFixed(2));
                $('#max-demand').text(response.max_demand.toFixed(2));
                $('#min-demand').text(response.min_demand.toFixed(2));

                // Render Yearly Total Demand (Bar Chart)
                Plotly.newPlot('yearly-demand-trend', [{
                    x: response.total_demand_per_year.map(d => d.year),
                    y: response.total_demand_per_year.map(d => d.total_demand),
                    type: 'bar',
                    marker: { color: 'steelblue' },
                }], {
                    xaxis: { title: 'Year' },
                    yaxis: { title: 'Total Demand (kWh)' },
                });

                // Render Monthly Average Demand (Line Chart)
                const monthlyData = response.monthly_demand.reduce((acc, d) => {
                    if (!acc[d.year]) acc[d.year] = { x: [], y: [], mode: 'lines+markers', name: `${d.year}` };
                    acc[d.year].x.push(d.month);
                    acc[d.year].y.push(d.value);
                    return acc;
                }, {});
                Plotly.newPlot('monthly-demand', Object.values(monthlyData), {
                    xaxis: { title: 'Month', dtick: 1 },
                    yaxis: { title: 'Average Demand (kWh)' },
                });

                // Render Day-of-Week Demand (Pie Chart)
                Plotly.newPlot('daily-demand', [{
                    labels: response.daily_demand.map(d => d.day_of_week),
                    values: response.daily_demand.map(d => d.value),
                    type: 'pie'
                }]);

                // Render Hourly Average Demand (Line Chart)
                Plotly.newPlot('hourly-demand', [{
                    x: response.hourly_demand.map(d => d.hour),
                    y: response.hourly_demand.map(d => d.avg_demand),
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: 'orange' },
                }], {
                    xaxis: { title: 'Hour of the Day', dtick: 1 },
                    yaxis: { title: 'Average Demand (kWh)' },
                });

                // Render Monthly Demand Variation (Box Plot)
                Plotly.newPlot('monthly-demand-variation', [{
                    x: response.monthly_demand.map(d => d.month),
                    y: response.monthly_demand.map(d => d.value),
                    type: 'box',
                    boxpoints: 'all',
                    jitter: 0.3
                }], {
                    xaxis: { title: 'Month', dtick: 1 },
                    yaxis: { title: 'Demand (kWh)' },
                });

                
                // Render Peak Demand Hours Analysis (Bar Chart)
                Plotly.newPlot('peak-demand-hours', [{
                    x: response.peak_demand_hours.map(d => d.hour),
                    y: response.peak_demand_hours.map(d => d.total_demand),
                    type: 'bar',
                    marker: { color: 'purple' },
                }], {
                    xaxis: { title: 'Hour of the Day', dtick: 1 },
                    yaxis: { title: 'Total Demand (kWh)' },
                });

                // Render Heatmap (Hourly vs. Day-of-Week)
                Plotly.newPlot('heatmap-hourly-day', [{
                    z: response.heatmap_hourly_day,
                    x: Array.from({ length: 24 }, (_, i) => i),
                    y: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    type: 'heatmap',
                    colorscale: 'Viridis',
                }], {
                    xaxis: { title: 'Hour of the Day' },
                    yaxis: { title: 'Day of the Week' },
                });


            } catch (error) {
                console.error("Data processing error:", error);
                $('#error-message').text('Error processing demand data. Please try again later.').show();
            }
        },
        error: function (xhr, status, error) {
            console.error("AJAX Error:", error);
            $('#error-message').text('Failed to fetch demand data. Please try again later.').show();
            setTimeout(() => {
                $('#error-message').fadeOut();
            }, 5000);
        },
        complete: function () {
            $('#loading-spinner').hide();
        },
    });
}


// 10. END SECTION 10 EDA DEMAND


// 11. START SECTION 11 MLOPS DATA PREPROCESSING

function initializeDatasetPreviewTable() {
    const $table = $("#datasetPreviewTable");

    if ($table.length) {
        $table.DataTable({
            ajax: {
                url: `${baseUrl}/mlops_preprocessing/data`,
                dataSrc: "data",
                error: function (xhr, error, thrown) {
                    console.error("Error fetching dataset preview:", error);
                    alert("Failed to load dataset preview. Please try again later.");
                },
            },
            columns: [
                { data: "ds", title: "Date (DS)" },
                { data: "y", title: "Demand (Y)" },
                { data: "temp", title: "Temperature (°C)" },
                { data: "feelslike", title: "Feels Like (°C)" },
                { data: "humidity", title: "Humidity (%)" },
                { data: "windspeed", title: "Wind Speed (km/h)" },
                { data: "cloudcover", title: "Cloud Cover (%)" },
                { data: "solaradiation", title: "Solar Radiation (W/m²)" },
                { data: "precip", title: "Precipitation (mm)" },
                { data: "preciptype", title: "Precipitation Type" },
                { data: "date", title: "Date" },
                { data: "is_holiday", title: "Is Holiday" }
            ],
            ordering: false, // Prevent frontend reordering
            pageLength: 10,
            responsive: true,
            autoWidth: false,
            dom: "Bfrtip",
            buttons: ["copy", "csv", "excel", "pdf", "print"],
            language: {
                emptyTable: "No data available in the preview",
                loadingRecords: "Loading data...",
                zeroRecords: "No matching records found"
            },
        });
        console.log("Dataset preview table initialized successfully.");
    } else {
        console.warn("Dataset preview table element not found.");
    }
}


// Function to refresh dataset preview
function initializeDatasetPreviewRefresh() {
    const $refreshButton = $("#refreshPreview");

    if ($refreshButton.length) {
        $refreshButton.on("click", function () {
            const table = $("#datasetPreviewTable").DataTable();
            table.ajax.reload(null, false); // Reload without resetting pagination
        });
    } else {
        console.warn("Refresh button not found.");
    }
}

// 11. END SECTION 11 MLOPS DATA PREPROCESSING

// 12. START SECTION 12 MLOPS MODEL TRAINING

function renderParameters(data) {
    try {
        if (data && typeof data === "string") {
            return `<div style="white-space: pre-wrap;">${data}</div>`; // Preserve formatting
        }
        return "No Parameters";
    } catch (error) {
        console.error("Error rendering Parameters:", error, data);
        return "Invalid Parameters"; // Fallback message
    }
}



function parseJSON(response) {
    try {
        return typeof response === "string" ? JSON.parse(response) : response;
    } catch (error) {
        console.error("Error parsing JSON:", error);
        return null;
    }
}

function initializeValidationLogs() {
    const tableId = "#validation-logs-table";

    if ($(tableId).length) {
        $(tableId).DataTable({
            processing: true,
            serverSide: false,
            ajax: {
                url: `${baseUrl}/api/mlops/validation/logs`,
                type: "GET",
                dataSrc: function (json) {
                    console.log("Raw Validation Logs Data:", json.data); // Debugging the response
                    if (json && json.data) {
                        return json.data;
                    } else {
                        console.error("Invalid response format for validation logs:", json);
                        return [];
                    }
                },
                
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error("Error fetching validation logs:", errorThrown);
                    console.log("jqXHR", jqXHR);
                    console.log("textStatus", textStatus);

                    let message = "Unexpected error occurred.";
                    if (jqXHR.status === 404) {
                        message = "Validation logs not found (404).";
                    } else if (jqXHR.status === 400) {
                        console.log("jqXHR.responseJSON", jqXHR.responseJSON);
                        message = "Failed to load validation logs. Invalid data format.";
                    } else if (textStatus === "parsererror") {
                        console.log("jqXHR.responseText", jqXHR.responseText);
                        message = "Response data type error.";
                    } else if (textStatus === "timeout") {
                        message = "Connection timed out.";
                    }
                    alert(message);
                },
            },
            columns: [
                { data: "Model", title: "Model" },
                {
                    data: "Parameters",
                    title: "Parameters",
                    render: function (data) {
                        return `<pre style="white-space: pre-wrap;">${data}</pre>`;
                    },
                },
                { data: "MAE", title: "MAE" },
                { data: "MAPE", title: "MAPE" },
                { data: "RMSE", title: "RMSE" },
                { data: "RMSE", title: "RMSE" },
                { data: "r_squared", title: "R²" },
                { data: "RMSE", title: "RMSE" },
                { data: "Generated_At", title: "Generated At" },
            ],
            order: [[6, "desc"]],
        });
    }
}


function initializeTrainingLogs() {
    const tableId = "#training-logs-table";

    if ($(tableId).length) {
        $(tableId).DataTable({
            processing: true,
            serverSide: false,
            ajax: {
                url: `${baseUrl}/api/mlops/training/logs`,
                type: "GET",
                dataSrc: function (json) {
                    if (json && json.data) {
                        // Handle NaN or undefined values in the response
                        return json.data.map(row => {
                            for (const key in row) {
                                if (row[key] === null || row[key] === "NaN" || row[key] === undefined) {
                                    row[key] = "N/A"; // Replace invalid values with "N/A"
                                }
                            }
                            return row;
                        });
                    } else {
                        console.error("Invalid training logs response format:", json);
                        return [];
                    }
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error("Error fetching training logs:", errorThrown);
                    alert("Failed to load training logs.");
                },
            },
            columns: [
                { data: "Model", title: "Model" },
                { data: "Iteration", title: "Iteration" },
                {
                    data: "Parameters",
                    title: "Parameters",
                    render: function (data) {
                        return `<pre style="white-space: pre-wrap;">${data || "N/A"}</pre>`;
                    },
                },
                { data: "MAE", title: "MAE", render: data => data || "N/A" },
                { data: "MAPE", title: "MAPE", render: data => data || "N/A" },
                { data: "RMSE", title: "RMSE", render: data => data || "N/A" },
                { data: "MSE", title: "MSE", render: data => data || "N/A" },
                { data: "R²", title: "R²", render: data => data || "N/A" },
                { data: "MBE", title: "MBE", render: data => data || "N/A" },
                { data: "Generated_At", title: "Generated At", render: data => data || "N/A" },
            ],
            order: [[7, "desc"]], // Sort by "Generated At"
            pageLength: 10,
            responsive: true,
        });
    }
}

function initializeRefreshTrainingLogsButton() {
    const refreshButtonId = "#refreshTrainingLogs";

    $(refreshButtonId).on("click", function () {
        const table = $("#training-logs-table").DataTable();
        if (table) {
            table.ajax.reload(null, false);
            alert("Training logs refreshed.");
        }
    });
}



function initializeRefreshButtons() {
    // Refresh Training Logs
    const $refreshTraining = $("#refreshTrainingLogs");
    if ($refreshTraining.length) {
        $refreshTraining.on("click", function () {
            const table = $("#training-logs-table").DataTable();
            if (table) {
                table.ajax.reload(null, false);
                alert("Training logs have been refreshed.");
            }
        });
    }

    // Refresh Validation Logs
    const $refreshValidation = $("#refreshValidationLogs");
    if ($refreshValidation.length) {
        $refreshValidation.on("click", function () {
            const table = $("#validation-logs-table").DataTable();
            if (table) {
                table.ajax.reload(null, false);
                alert("Validation logs have been refreshed.");
            }
        });
    }
}


// 12. END SECTION 12 MLOPS MODEL TRAINING
// START SECTION 13 MLOPS MODEL EVALUATION


function initializeMetricsSummaryTable() {
    const tableId = '#metrics-summary';

    if ($(tableId).length) {
        $(tableId).DataTable({
            ajax: {
                url: `${baseUrl}/api/mlops_predictionevaluation`,
                type: 'GET',
                dataSrc: response => {
                    if (!response || !response.metrics_summary) {
                        alert('No metrics summary data available.');
                        return [];
                    }
                    return response.metrics_summary;
                },
                error: function () {
                    alert('Error fetching metrics summary data.');
                },
            },
            columns: [
                { data: 'model', title: 'Model' },
                { data: 'mae', title: 'MAE', render: data => parseFloat(data).toFixed(2) },
                { data: 'mape', title: 'MAPE', render: data => parseFloat(data).toFixed(2) },
                { data: 'rmse', title: 'RMSE', render: data => parseFloat(data).toFixed(2) },
                { data: 'r_squared', title: 'R²', render: data => parseFloat(data).toFixed(2) },
                { data: 'mbe', title: 'MBE', render: data => parseFloat(data).toFixed(2) },
                { data: 'parameters', title: 'Parameters' }
            ],
            pageLength: 10,
            responsive: true,
        });
    }
}

/**
 * Initialize Prediction Trends Chart
 */
function initializePredictionTrendsChart() {
    $.ajax({
        url: `${baseUrl}/api/mlops_predictionevaluation`,
        type: 'GET',
        success: function (response) {
            const data = response.prediction_comparison || [];
            const traces = {};

            data.forEach(item => {
                if (!traces[item.model]) traces[item.model] = { x: [], y: [] };
                traces[item.model].x.push(item.ds);
                traces[item.model].y.push(item.prophet_predicted || 0);
            });

            const plotlyTraces = Object.keys(traces).map(model => ({
                x: traces[model].x,
                y: traces[model].y,
                mode: 'lines',
                name: model,
            }));

            Plotly.newPlot('predictionTrendsChart', plotlyTraces, {
                title: 'Prediction Trends',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Predicted Values' },
            });
        },
        error: function () {
            alert('Failed to fetch prediction trends data.');
        }
    });
}


function renderPredictionComparisonChart(data) {
    const chartContainer = document.getElementById('predictionComparisonChart');
    if (!chartContainer) {
        console.error(`Chart container with ID predictionComparisonChart not found.`);
        return;
    }

    const groupedData = data.reduce((acc, curr) => {
        if (!acc[curr.model]) acc[curr.model] = { x: [], y: [] };
        acc[curr.model].x.push(curr.ds);
        acc[curr.model].y.push(curr.y || 0); // Replace null/undefined with 0
        return acc;
    }, {});

    const traces = Object.keys(groupedData).map(model => ({
        x: groupedData[model].x,
        y: groupedData[model].y,
        mode: 'lines',
        name: model,
    }));

    Plotly.newPlot(chartContainer, traces, {
        title: 'Future Prediction Comparison',
        xaxis: { title: 'Date', type: 'date' },
        yaxis: { title: 'Predictions' },
    });
}


function initializePredictionComparisonTable() {
    const tableId = '#prediction-comparison';
    const chartId = 'predictionComparisonChart';

    if ($(tableId).length) {
        $(tableId).DataTable({
            processing: true,
            serverSide: false,
            ajax: {
                url: `${baseUrl}/api/mlops_predictionevaluation`,
                type: 'GET',
                dataSrc: response => {
                    console.log('API Response Data:', response.prediction_comparison);

                    // Debug the API response to ensure 'prediction_comparison' exists
                    if (!response || !response.prediction_comparison) {
                        console.warn('No prediction comparison data available.');
                        alert('No prediction comparison data available.');
                        return [];
                    }
                    // Log the response for debugging
                    console.log('Prediction Comparison Data:', response.prediction_comparison);
                    renderPredictionComparisonChart(response.prediction_comparison);
                    return response.prediction_comparison;
                },
                error: function () {
                    alert('Failed to fetch prediction comparison data.');
                },
            },
            columns: [
                { data: 'ds', title: 'Date' }, // Maps the 'ds' field to the Date column
                {
                    data: 'y', 
                    title: 'Prediction', 
                    render: data => (data !== null && !isNaN(data)) ? parseFloat(data).toFixed(2) : '0' // Render 'y' correctly
                },
                { data: 'model', title: 'Model' }, // Maps the 'model' field to the Model column
            ],
        });
    }
}



function initializeMetricsOverviewChart() {
    const chartId = 'metricsSummaryChart';
    const chartContainer = document.getElementById(chartId);

    $.ajax({
        url: `${baseUrl}/api/mlops_predictionevaluation`,
        type: 'GET',
        success: function (response) {
            const metrics = response.chart_data;
            const models = metrics.map(item => item.model);
            const mae = metrics.map(item => parseFloat(item.mae) || 0);
            const rmse = metrics.map(item => parseFloat(item.rmse) || 0);
            const rSquared = metrics.map(item => parseFloat(item.r_squared) || 0);

            const traces = [
                { x: models, y: mae, name: 'MAE', type: 'bar', marker: { color: '#4e73df' } },
                { x: models, y: rmse, name: 'RMSE', type: 'bar', marker: { color: '#36b9cc' } },
                { x: models, y: rSquared, name: 'R²', type: 'bar', marker: { color: '#1cc88a' } },
            ];

            Plotly.newPlot(chartContainer, traces, {
                title: 'Metrics Overview Across Models',
                barmode: 'group',
                xaxis: { title: 'Models', tickangle: -45 },
                yaxis: { title: 'Metric Values', zeroline: true },
            });
        },
        error: function () {
            alert('Failed to load metrics overview data.');
        },
    });
}



function fetchAndUpdateBestModelUI() {
    $.ajax({
        url: `${baseUrl}/api/mlops_predictionevaluation`,
        type: 'GET',
        success: function (response) {
            const bestModel = response.metrics_summary.reduce((best, current) => {
                const rank = current.mae + (1 - current.r_squared);
                return rank < best.rank ? { ...current, rank } : best;
            }, { rank: Infinity });

            $('#best-model-name').text(bestModel.model || '--');
            $('#best-model-mae').text(bestModel.mae.toFixed(2) || '--');
            $('#best-model-r2').text(bestModel.r_squared.toFixed(2) || '--');
        },
        error: function () {
            alert('Failed to fetch the best model data.');
        }
    });
}


function initializeModelComparisonChart() {
    $.ajax({
        url: `${baseUrl}/api/mlops_predictionevaluation`,
        type: 'GET',
        success: function (response) {
            const metrics = response.metrics_summary;
            const models = metrics.map(item => item.model);
            const mae = metrics.map(item => item.mae);
            const r_squared = metrics.map(item => item.r_squared);
            const rmse = metrics.map(item => item.rmse);

            const traces = [
                { x: models, y: mae, name: 'MAE', type: 'bar' },
                { x: models, y: r_squared, name: 'R²', type: 'bar' },
                { x: models, y: rmse, name: 'RMSE', type: 'bar' },
            ];

            Plotly.newPlot('modelComparisonChart', traces, {
                title: 'Model Metrics Comparison',
                barmode: 'group',
                xaxis: { title: 'Models' },
                yaxis: { title: 'Metrics' },
            });
        },
        error: function () {
            alert('Failed to load model comparison data.');
        }
    });
}
function renderRadarChart(metrics) {
    const chartContainer = document.getElementById('metricsRadarChart');

    // Filter for only three specific models
    const modelsToInclude = ['Darts Theta', 'Prophet', 'GradientBoostingRegressor'];
    const filteredMetrics = metrics
        .filter(metric => modelsToInclude.includes(metric.model))
        .reduce((unique, item) => {
            // Remove duplicates by checking for unique models
            if (!unique.some(metric => metric.model === item.model)) {
                unique.push(item);
            }
            return unique;
        }, []);

    // Prepare labels and datasets
    const labels = ['MAE', 'MAPE', 'RMSE', 'R²', 'MBE'];
    const datasets = filteredMetrics.map(metric => ({
        label: metric.model,
        data: [
            parseFloat(metric.mae) || 0,
            parseFloat(metric.mape) || 0,
            parseFloat(metric.rmse) || 0,
            parseFloat(metric.r_squared) || 0,
            parseFloat(metric.mbe) || 0
        ],
        fill: true,
        backgroundColor: getRandomColorWithAlpha(0.4),
        borderColor: getRandomColorWithAlpha(1),
        borderWidth: 2
    }));

    // Render the Radar Chart
    new Chart(chartContainer, {
        type: 'radar',
        data: {
            labels,
            datasets,
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
            },
            scales: {
                r: {
                    suggestedMin: 0,
                    suggestedMax: 7000, // Adjust based on expected range
                    angleLines: { color: '#ddd' }, // Styling
                    grid: { color: '#ccc' },
                },
            },
        },
    });
}


// Utility function for generating random colors with transparency
function getRandomColorWithAlpha(alpha) {
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Fetch and render radar chart
function initializeRadarChart() {
    $.ajax({
        url: `${baseUrl}/api/mlops_predictionevaluation`,
        type: 'GET',
        success: function (response) {
            if (response && response.metrics_summary) {
                renderRadarChart(response.metrics_summary);
            } else {
                console.warn("Metrics summary data not available.");
            }
        },
        error: function (error) {
            console.error("Failed to fetch radar chart data:", error);
        },
    });
}


function renderHeatmap(metrics) {
    const chartContainer = document.getElementById('metricsHeatmap');
    const data = metrics.map(metric => [metric.model, metric.mae, metric.mape, metric.rmse, metric.r_squared]);
    const zData = data.map(row => row.slice(1)); // Exclude the model column

    Plotly.newPlot(chartContainer, [
        {
            z: zData,
            x: ['MAE', 'MAPE', 'RMSE', 'R²'],
            y: data.map(row => row[0]), // Model names
            type: 'heatmap',
            colorscale: 'Viridis',
        },
    ]);
}

