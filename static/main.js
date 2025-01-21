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

        if (currentPath === '/eda/holiday') {
            console.log('Initializing Holiday EDA...');
            fetchAndRenderHolidayData(); // Fetch and render EDA Holiday data
            
        }
    
        if (currentPath === '/eda/weather') {
            console.log('Initializing Weather EDA...');
            fetchAndRenderWeatherData(); // Fetch and render EDA Weather data
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
    
        if (currentPath === '/') {
            console.log('Initializing Dashboard...');
            fetchAndRenderDashboard(); // Fetch and render Dashboard data
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
    const menuSearchId = '#menuSearch'; // Input element for menu search

    if ($(menuSearchId).length) {
        $(menuSearchId).on('input', function () {
            const query = $(this).val().toLowerCase().trim();

            $('.sidebar .nav-item').each(function () {
                const text = $(this).text().toLowerCase();
                $(this).toggle(text.includes(query)); // Toggle visibility based on query match
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
function fetchAndRenderDashboard() {
    const dashboardContainerId = '#dashboard-container'; // Dashboard container
    const loaderId = '#dashboard-loader'; // Loader while fetching data

    if ($(dashboardContainerId).length) {
        $(loaderId).show(); // Show loader

        $.ajax({
            url: `${baseUrl}/api/dashboard`, // API endpoint for dashboard data
            type: 'GET',
            success: function (response) {
                if (response && response.data) {
                    const dashboardData = response.data;
                    renderDashboardData(dashboardData); // Populate dashboard widgets
                } else {
                    console.error("Invalid dashboard response:", response);
                    alert("Failed to fetch dashboard data.");
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                console.error('Error fetching dashboard data:', {
                    status: jqXHR.status,
                    responseText: jqXHR.responseText,
                    textStatus,
                    errorThrown,
                });
                alert("Unable to load dashboard data.");
            },
            complete: function () {
                $(loaderId).hide(); // Hide loader after fetching
            },
        });
    }
}

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
});

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





// EDA DEMAND
// Add this to handle demand EDA dynamically
/*document.addEventListener("DOMContentLoaded", function () {

    // Event Listener for EDA Demand Navigation
    const edaDemandNav = document.querySelector("a[href='/eda/demand']");

    if (edaDemandNav) {
        edaDemandNav.addEventListener("click", function (e) {
            e.preventDefault(); // Prevent default navigation behavior
            fetch('/eda/demand')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.text(); // Return the HTML content
                })
                .then(html => {
                    // Dynamically replace the content in the "#content-wrapper"
                    const contentWrapper = document.querySelector("#content-wrapper");
                    if (contentWrapper) {
                        contentWrapper.innerHTML = html;

                        // After content loads, initialize Plotly charts
                        const monthlyAvgDataElement = document.getElementById("monthly-avg-data");
                        const hourlyAvgDataElement = document.getElementById("hourly-avg-data");
                        const heatmapDataElement = document.getElementById("heatmap-data");

                        // Check for each chart's data and render it dynamically
                        if (monthlyAvgDataElement) {
                            const monthlyAvgData = JSON.parse(monthlyAvgDataElement.textContent);
                            Plotly.newPlot('monthly-avg-chart', monthlyAvgData.data, monthlyAvgData.layout);
                        }

                        if (hourlyAvgDataElement) {
                            const hourlyAvgData = JSON.parse(hourlyAvgDataElement.textContent);
                            Plotly.newPlot('hourly-avg-chart', hourlyAvgData.data, hourlyAvgData.layout);
                        }

                        if (heatmapDataElement) {
                            const heatmapData = JSON.parse(heatmapDataElement.textContent);
                            Plotly.newPlot('heatmap-chart', heatmapData.data, heatmapData.layout);
                        }
                    } else {
                        console.error("Error: '#content-wrapper' not found.");
                    }
                })
                .catch(error => {
                    console.error('Error loading Demand EDA page:', error);
                    alert('Failed to load the Demand EDA page. Please try again later.');
                });
        });
    }
});*/


// 8. END SECTION 8 EDA DEMAND