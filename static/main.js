// At the very beginning of main.js:
const baseUrl = 'https://www.electforecast.de'; // Replace with your actual URL
//const baseUrl = "http://localhost:8000";

if (typeof $ === 'undefined') {
    console.error('jQuery is not loaded. Skipping script execution.');
    alert("The application requires jQuery to function properly.");
} else {
    console.log('jQuery loaded successfully.');

    $(document).ready(function () {
        initializeAutoUpdate(); // Auto-update the "Last Updated" section
        initializeTables(); // Initialize all DataTables (demand, weather, holidays)
        initializeSearchInput(); // Search functionality for input fields
        initializeMenuSearch(); // Sidebar menu search functionality
    });

    //1. START SECTION 1 MENU AND SEARCH
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
    function initializeSearchInput() {
        const searchInputId = '#searchInput';
    
        if ($(searchInputId).length) {
            let debounceTimer;
            $(searchInputId).on('input', function () {
                clearTimeout(debounceTimer);
                const searchValue = $(this).val().trim();
    
                debounceTimer = setTimeout(() => {
                    if ($('#hourly-demand-table').length) {
                        $('#hourly-demand-table').DataTable().search(searchValue).draw();
                    }
                    if ($('#hourly-weather-table').length) {
                        $('#hourly-weather-table').DataTable().search(searchValue).draw();
                    }
                    if ($('#holidays-table').length) {
                        $('#holidays-table').DataTable().search(searchValue).draw();
                    }
                }, 300);
            });
        }
    }     
    //1. END SECTION 1 MENU AND SEARCH

    //2. START SECTION 2 DASHBOARD

    //2. END SECTION 2 DASHBOARD

    //3. START SECTION 3 HOURLYDEMAND
    function initializeHourlyDemandTable() {
        const tableId = '#hourly-demand-table';
    
        if ($.fn.DataTable.isDataTable(tableId)) {
            $(tableId).DataTable().destroy();
        }
    
        $(tableId).DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: `${baseUrl}/api/hourlydemand`,
                type: 'GET',
                dataSrc: function (json) {
                    if (!json.data) {
                        console.error("Invalid JSON response:", json);
                        return [];
                    }
                    return json.data;
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error('Error fetching data:', {
                        status: jqXHR.status,
                        responseText: jqXHR.responseText,
                        textStatus,
                        errorThrown,
                    });
                    alert("Failed to load hourly demand data.");
                },
            },
            columns: [
                { data: 'time', title: 'Datetime' },
                { data: 'value', title: 'Demand (kWh)' },
            ],
            order: [[0, 'desc']],
            pageLength: 10,
        });
    }    
    //3. END SECTION 3 HOURLYDEMAND

    //4. START SECTION 4 HOURLYWEATHER
    function initializeHourlyWeatherTable() {
        const tableId = '#hourly-weather-table';
    
        if ($.fn.DataTable.isDataTable(tableId)) {
            $(tableId).DataTable().destroy();
        }
    
        $(tableId).DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: `${baseUrl}/api/hourlyweather`,
                type: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                dataSrc: function (json) {
                    if (!json.data) {
                        console.error("Invalid Hourly Weather JSON response:", json);
                        return [];
                    }
                    return json.data;
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error('Error fetching hourly weather data:', {
                        status: jqXHR.status,
                        responseText: jqXHR.responseText,
                        textStatus: textStatus,
                        errorThrown: errorThrown,
                    });
                }
            },
            columns: [
                { data: 'datetime', title: 'Datetime' },
                { data: 'temp', title: 'Temperature (°C)' },
                { data: 'feelslike', title: 'Feels Like (°C)' },
                { data: 'humidity', title: 'Humidity (%)' },
                { data: 'windspeed', title: 'Wind Speed (km/h)' },
                { data: 'cloudcover', title: 'Cloud Cover (%)' },
                { data: 'solaradiation', title: 'Solar Radiation (W/m²)' },
                { data: 'precip', title: 'Precipitation (mm)' },
                { data: 'preciptype', title: 'Precipitation Type' }
            ],
            order: [[0, 'desc']],
            pageLength: 10,
        });
    }
    //4. END SECTION 4 HOURLYWEATHER

    //5. START SECTION 5 HOLIDAYS
    function initializeHolidaysTable() {
        const tableId = '#holidays-table';
    
        if ($.fn.DataTable.isDataTable(tableId)) {
            $(tableId).DataTable().destroy();
        }
    
        $(tableId).DataTable({
            serverSide: true,
            processing: true,
            ajax: {
                url: `${baseUrl}/api/holidays`,
                type: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                dataSrc: function (json) {
                    if (!json.data) {
                        console.error("Invalid Holidays JSON response:", json);
                        return [];
                    }
                    return json.data;
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.error('Error fetching holidays data:', {
                        status: jqXHR.status,
                        responseText: jqXHR.responseText,
                        textStatus: textStatus,
                        errorThrown: errorThrown,
                    });
                }
            },
            columns: [
                { data: 'date', title: 'Date' },
                { data: 'name', title: 'Holiday Name' }
            ],
            order: [[0, 'desc']],
            pageLength: 10,
        });
    }
    //5. END SECTION 5 HOLIDAYS

    //6. START SECTION 6 LAST UPDATED
    function updateLastUpdated() {
        $.ajax({
            url: `${baseUrl}/api/lastUpdated`,
            type: 'GET',
            success: ({ lastUpdatedDemand, lastUpdatedWeather, lastUpdatedHoliday }) => {
                $('#last-updated-demand em').text(
                    lastUpdatedDemand ? `Data last updated as of ${lastUpdatedDemand}` : 'No updates available.'
                );
                $('#last-updated-weather em').text(
                    lastUpdatedWeather ? `Data last updated as of ${lastUpdatedWeather}` : 'No updates available.'
                );
                $('#last-updated-holiday em').text(
                    lastUpdatedHoliday ? `Data last updated as of ${lastUpdatedHoliday}` : 'No updates available.'
                );
            },
            error: ({ status, responseText, statusText }) => {
                console.error('Error updating last updated data:', {
                    status,
                    responseText,
                    statusText,
                });
                alert("Unable to fetch 'Last Updated' data. Please try refreshing the page.");
            },
        });
    }
    
    
    function initializeAutoUpdate() {
        updateLastUpdated(); // Initial call
        setInterval(updateLastUpdated, 60000); // Repeat every 60 seconds
    }
 
    //6. END SECTION 6 LAST UPDATED

    //7. START SECTION 7 INITIALIZE TABLES
    function initializeTables() {
        if ($('#hourly-demand-table').length) {
            initializeHourlyDemandTable();
        }
        if ($('#hourly-weather-table').length) {
            initializeHourlyWeatherTable();
        }
        if ($('#holidays-table').length) {
            initializeHolidaysTable();
        }
    }
    //7. END SECTION 7 INITIALIZE TABLES
}
